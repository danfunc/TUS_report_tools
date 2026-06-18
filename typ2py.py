import argparse
import sys
import re
import numpy as np

# ==========================================
# 1. 字句解析 (Lexer / Tokenizer)
# ==========================================
TOKEN_SPEC = [
    ('NUMBER',   r'\d+(\.\d*)?'),
    ('SYMBOL',   r'[A-Za-z][A-Za-z0-9]*'),
    ('OP',       r'[+\-*/^=_]'),
    ('SKIP',     r'[ \t]+'),
    ('MISMATCH', r'.'),
]
TOKEN_REGEX = re.compile('|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in TOKEN_SPEC))

def extract_variables(nodes, seen=None, local_vars=None):
    """
    Auto-Detect機能。
    SIGMAのイテレータ変数(i)はローカルスコープとして扱い、グローバル引数からは除外する。
    """
    if seen is None: seen = []
    if local_vars is None: local_vars = set()
    
    for n in nodes:
        if n['type'] == 'SYMBOL':
            sym = n['value']
            # ローカル変数(iなど)でなければグローバルとして登録
            if sym not in FUNCTION_SYMBOLS and sym not in BUILTIN_VALUE_SYMBOLS and sym not in seen and sym not in local_vars:
                seen.append(sym)
        elif n['type'] == 'GROUP':
            extract_variables(n['children'], seen, local_vars)
        elif n['type'] == 'SIGMA':
            # 1. SIGMA内部だけの新しいローカルスコープを作成
            new_locals = local_vars.copy()
            new_locals.add(n['iter_var'])
            
            # 2. 終端値(n)の探索（ここは外側のスコープ）
            extract_variables([n['end_node']], seen, local_vars)
            
            # 3. 評価式(i^2)の探索（ここはローカルスコープを適用）
            extract_variables(n['body'], seen, new_locals)
            
    return seen




def tokenize(text):
    tokens = []
    for match in TOKEN_REGEX.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == 'SKIP': continue
        if kind == 'MISMATCH': raise SyntaxError(f"Lexer Error: 予期しない文字 '{value}'")
        tokens.append({'type': kind, 'value': value})
    return tokens

# ==========================================
# 2. 構文解析 (Parser / AST Builder)
# ==========================================
def build_ast(expr):
    stack = [[]]
    buffer = ""
    def flush():
        nonlocal buffer
        if buffer.strip():
            stack[-1].extend(tokenize(buffer))
        buffer = ""

    for i, char in enumerate(expr):
        if char == '(' or  char == '{':
            flush()
            stack.append([])
        elif char == ')' or char == '}':
            flush()
            if len(stack) == 1: raise SyntaxError(f"Parser Error: 位置 {i} に予期しない閉じ括弧")
            group_children = stack.pop()
            stack[-1].append({'type': 'GROUP', 'children': group_children})
        else:
            buffer += char

    flush()
    if len(stack) > 1: raise SyntaxError("Parser Error: 閉じられていない開き括弧")
    return stack[0]

def resolve_subscripts(nodes):
    """
    optimize_summation の後に実行するパス。
    残された '_' 演算子を検知し、前後のノードを結合して1つのSYMBOL変数に再編する。
    """
    new_nodes = []
    i = 0
    while i < len(nodes):
        # 境界チェックとパターンマッチ: [SYMBOL] + '_' + [SYMBOL または NUMBER]
        if (i + 2 < len(nodes) and 
            nodes[i]['type'] == 'SYMBOL' and 
            nodes[i+1]['type'] == 'OP' and nodes[i+1]['value'] == '_' and 
            nodes[i+2]['type'] in ('SYMBOL', 'NUMBER')):
            
            # 3つのノードを1つの変数名として結合
            combined_name = f"{nodes[i]['value']}_{nodes[i+2]['value']}"
            new_nodes.append({'type': 'SYMBOL', 'value': combined_name})
            i += 3
        else:
            # 括弧グループの中身も再帰的に解決
            if nodes[i]['type'] == 'GROUP':
                nodes[i]['children'] = resolve_subscripts(nodes[i]['children'])
            # SIGMA内部（終端値・body）も再帰的に解決
            elif nodes[i]['type'] == 'SIGMA':
                nodes[i]['end_node'] = resolve_subscripts([nodes[i]['end_node']])[0]
                nodes[i]['body'] = resolve_subscripts(nodes[i]['body'])
            new_nodes.append(nodes[i])
            i += 1
            
    return new_nodes

def optimize_summation(nodes):
    """
    ASTを巡回し、Typstの 'sum' 記法を検知して特殊な 'SIGMA' ノードに変換する。
    期待するパターン: sum _ (i = 1) ^ n ( body )
    """
    new_nodes = []
    i = 0
    
    while i < len(nodes):
        current = nodes[i]
        
        # 'sum' シンボルを発見した場合、ルックアヘッド（先読み）で後続のノード構造を検証する
        if current['type'] == 'SYMBOL' and current['value'] in ('sum', 'Sigma'):
            try:
                # 1. 下付き文字 (初期化ブロック) '_'
                assert nodes[i+1]['value'] == '_', "sumの後には '_' が必要です"
                init_group = nodes[i+2]
                assert init_group['type'] == 'GROUP', "初期化式は括弧で囲む必要があります"
                
                # 初期化式 "i = 1" からイテレータ変数と初期値を抽出
                # (トークン分割されているため [SYMBOL('i'), OP('='), NUMBER('1')] となる)
                iter_var = init_group['children'][0]['value']
                start_val = init_group['children'][2]['value']
                
                # 2. 上付き文字 (終端ブロック) '^'
                assert nodes[i+3]['value'] == '^', "初期化式の後には '^' が必要です"
                end_node = nodes[i+4] # これは 'n' のようなSYMBOLか、GROUPになり得る
                
                # 3. 評価される遅延ブロック (Body)
                body_group = nodes[i+5]
                assert body_group['type'] == 'GROUP', "総和の対象(body)は括弧で囲む必要があります"
                
                # 構造が完全に一致した場合、これら6つのノードを1つの 'SIGMA' ノードに圧縮する
                new_nodes.append({
                    'type': 'SIGMA',
                    'iter_var': iter_var,
                    'start_val': start_val,
                    'end_node': end_node,
                    'body': optimize_summation(body_group['children']) # body内部も再帰的に最適化
                })
                
                # パターンマッチした6つのノード分、インデックスを飛ばす
                i += 6
                continue
                
            except (IndexError, AssertionError) as e:
                # 構文が完全でない場合はエラーを吐いてコンパイルを落とす
                raise SyntaxError(f"Sigma(sum)の構文解析エラー: {e}\n正しい形式: sum_(i=1)^n (expr)")
        
        # 通常のグループであれば中身を再帰的に処理
        if current['type'] == 'GROUP':
            current['children'] = optimize_summation(current['children'])
            
        new_nodes.append(current)
        i += 1
        
    return new_nodes

# ==========================================
# 3. 意味解析 & 最適化 (Semantic Analyzer)
# ==========================================
FUNCTION_SYMBOLS = {'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'sum'}
BUILTIN_VALUE_SYMBOLS = {'pi', 'e'}

class SemanticError(Exception): pass

def analyze_semantics(nodes, variables, local_vars=None):
    """
    意味解析と暗黙の乗算。
    SIGMA内部でのみ有効な変数を解決し、SIGMA自体も1つの「値」として扱う。
    """
    if local_vars is None: local_vars = set()
    
    # 解決可能な値＝グローバル変数 ＋ ローカル変数 ＋ 組み込み定数
    VALUE_SYMBOLS = BUILTIN_VALUE_SYMBOLS.union(variables).union(local_vars)
    new_nodes = []
    
    for i in range(len(nodes)):
        current = nodes[i]
        
        if current['type'] == 'SYMBOL':
            sym = current['value']
            if sym in FUNCTION_SYMBOLS:
                current['symbol_role'] = 'FUNCTION'
            elif sym in VALUE_SYMBOLS:
                current['symbol_role'] = 'VALUE'
            else:
                raise SemanticError(f"Semantic Error: 未定義のシンボル '{sym}'。")
                
        elif current['type'] == 'GROUP':
            current['children'] = analyze_semantics(current['children'], variables, local_vars)
            
        elif current['type'] == 'SIGMA':
            # SIGMA特有のスコープ解決
            new_locals = local_vars.copy()
            new_locals.add(current['iter_var'])
            
            # end_nodeを解析して戻す
            analyzed_end = analyze_semantics([current['end_node']], variables, local_vars)
            current['end_node'] = analyzed_end[0]
            
            # bodyをローカルスコープ付きで解析して戻す
            current['body'] = analyze_semantics(current['body'], variables, new_locals)
            
        new_nodes.append(current)
        
        if i + 1 >= len(nodes): continue
        next_node = nodes[i + 1]
        
        # 暗黙の乗算判定（SIGMA自体も1つの値の塊として扱う）
        current_is_value = (current['type'] in ('NUMBER', 'GROUP', 'SIGMA') or 
                           (current['type'] == 'SYMBOL' and current.get('symbol_role') == 'VALUE'))
        next_is_value = next_node['type'] in ('NUMBER', 'SYMBOL', 'GROUP', 'SIGMA')
        
        if current_is_value and next_is_value:
            new_nodes.append({'type': 'OP', 'value': '*'})
            
    return new_nodes

# ==========================================
# 4. コード生成 (Code Generator)
# ==========================================
def generate_code(nodes):
    """
    意味解析済みのASTを巡回し、実行可能なPython(NumPy)コード文字列を生成する。
    """
    code = ""
    for node in nodes:
        if node['type'] == 'GROUP':
            # 括弧グループは再帰的に生成して () で囲む
            code += f"({generate_code(node['children'])})"
        elif node['type'] == 'NUMBER':
            code += node['value']
        elif node['type'] == 'OP':
            # Typstの累乗(^)をPython(**)に変換
            val = node['value']
            code += f" ** " if val == '^' else f" {val} "
        elif node['type'] == 'SYMBOL':
            sym = node['value']
            role = node.get('symbol_role')
            if role == 'FUNCTION':
                if sym == 'log': code += "np.log10"
                elif sym == 'ln': code += "np.log"
                else: code += f"np.{sym}"
            elif role == 'VALUE':
                if sym in BUILTIN_VALUE_SYMBOLS:
                    code += f"np.{sym}"
                else:
                    code += sym # ユーザー定義変数はそのまま出力
        elif node['type'] == 'SIGMA':
            # 抽出したスコープ変数と境界値
            iter_var = node['iter_var']
            start_code = node['start_val']
            
            # end_node は単一の変数の場合もあれば、数式ブロックの場合もあるため再帰的に生成
            end_code = generate_code([node['end_node']])
            
            # Body（総和の中身）を再帰的に生成
            body_code = generate_code(node['body'])
            
            # Pythonのジェネレータ式に変換（rangeは終端を含まないため + 1 が必要）
            # int() キャストを入れることで浮動小数点数の境界値エラーを防ぐ
            code += f"sum({body_code} for {iter_var} in range(int({start_code}), int({end_code}) + 1))"
    return code

# ==========================================
# 5. 統合トランスパイラ (The Transpiler)
# ==========================================
class TypstMathTranspiler:
    def transpile(self, typst_expr, variables):
        try:
            # フロントエンドからバックエンドへの一連のパスを実行
            ast = build_ast(typst_expr)
            ast_semantic = analyze_semantics(ast, variables)
            py_code_body = generate_code(ast_semantic)
            
            # lambda式として組み立て
            args_str = ", ".join(variables)
            lambda_str = f"lambda {args_str}: {py_code_body}"
            
            print(f"[Compiled] Typst expr:    {typst_expr}")
            print(f"           Python expr-> lambda({args_str}):")
            print(f"                            {py_code_body}")
            
            # 動的コンパイル
            return eval(lambda_str, {"np": np})
            
        except (SyntaxError, SemanticError) as e:
            print(f"[Compile Failed] {e}")
            return None


# ==========================================
# CLI (コマンドラインインターフェース) 実装
# ==========================================
# Auto-Detect用のextract_variablesはファイル先頭で定義済み(SIGMA対応版)。
# ここに同名のSIGMA非対応版が重複定義されており、Pythonの仕様上それが
# 後勝ちで有効になっていたため、sum内部だけの変数(例: nやx_1)が
# Auto-Detectで検出できなくなっていた。重複定義を削除し、先頭の正しい版を使う。

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typst Math to Python Transpiler")
    parser.add_argument("expr", type=str, help="Typstの数式表現 (例: '20 log( K / sqrt(1 + (2 pi f T)^2) )')")
    parser.add_argument("-v", "--vars", nargs="*", default=None, help="明示的に指定する変数リスト (例: -v f K T)")
    parser.add_argument("-e", "--eval", nargs="*", type=float, default=None, help="評価用の数値リスト (例: -e 1000 1.0 0.001)")
    
    args = parser.parse_args()
    
    try:
        # 1. パース (AST構築)
        ast = build_ast(args.expr)

        # 1.1 AST内部のSigmaパターンについて処理
        ast = optimize_summation(ast)

        # 1.2 Sigmaパターンではない_演算子をシンボル結合演算子として処理
        ast = resolve_subscripts(ast)
        
        # 2. 変数の決定 (指定がない場合はAuto-Detect)
        if args.vars is not None:
            variables = args.vars
            detect_mode = "Manual"
        else:
            variables = extract_variables(ast)
            detect_mode = "Auto-Detect"
            
        # 3. 意味解析 & コード生成
        ast_semantic = analyze_semantics(ast, variables)
        py_code_body = generate_code(ast_semantic)
        
        args_str = ", ".join(variables)
        lambda_str = f"lambda {args_str}: {py_code_body}"
        
        # 4. 出力フォーマット
        print("="*50)
        print(f"【Typst Expression】\n  {args.expr}")
        print("-" * 50)
        print(f"【Variables】 ({detect_mode})\n  {variables}")
        print("-" * 50)
        print(f"【Python Lambda】\n  {lambda_str}")
        print("="*50)
        
        # 5. (オプション) 評価実行
        if args.eval:
            if len(args.eval) != len(variables):
                print(f"[Error] 変数の数({len(variables)})と引数の数({len(args.eval)})が一致しません。", file=sys.stderr)
                sys.exit(1)
                
            func = eval(lambda_str, {"np": np})
            result = func(*args.eval)
            
            eval_str = ", ".join([f"{var}={val}" for var, val in zip(variables, args.eval)])
            print(f"【Execution Result】\n  f({eval_str}) -> {result:.4f}")
            print("="*50)

    except (SyntaxError, SemanticError) as e:
        print(f"\n[Compile Failed] {e}", file=sys.stderr)
        sys.exit(1)