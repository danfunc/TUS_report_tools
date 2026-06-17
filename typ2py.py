import argparse
import sys
import re
import numpy as np

# ==========================================
# 1. 字句解析 (Lexer / Tokenizer)
# ==========================================
TOKEN_SPEC = [
    ('NUMBER',   r'\d+(\.\d*)?'),
    ('SYMBOL',   r'[A-Za-z_][A-Za-z0-9_]*'),
    ('OP',       r'[+\-*/^=]'),
    ('SKIP',     r'[ \t]+'),
    ('MISMATCH', r'.'),
]
TOKEN_REGEX = re.compile('|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in TOKEN_SPEC))

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

# ==========================================
# 3. 意味解析 & 最適化 (Semantic Analyzer)
# ==========================================
FUNCTION_SYMBOLS = {'sin', 'cos', 'tan', 'log', 'ln', 'sqrt'}
BUILTIN_VALUE_SYMBOLS = {'pi', 'e'}

class SemanticError(Exception): pass

def analyze_semantics(nodes, variables):
    VALUE_SYMBOLS = BUILTIN_VALUE_SYMBOLS.union(variables)
    new_nodes = []
    
    for i in range(len(nodes)):
        current = nodes[i]
        
        # シンボルの名前解決
        if current['type'] == 'SYMBOL':
            sym = current['value']
            if sym in FUNCTION_SYMBOLS:
                current['symbol_role'] = 'FUNCTION'
            elif sym in VALUE_SYMBOLS:
                current['symbol_role'] = 'VALUE'
            else:
                raise SemanticError(f"Semantic Error: 未定義のシンボル '{sym}'。変数なら variables に追加しろ。")
                
        elif current['type'] == 'GROUP':
            current['children'] = analyze_semantics(current['children'], variables)
            
        new_nodes.append(current)
        
        if i + 1 >= len(nodes): continue
        next_node = nodes[i + 1]
        
        # 暗黙の乗算の論理的挿入
        current_is_value = (current['type'] in ('NUMBER', 'GROUP') or 
                           (current['type'] == 'SYMBOL' and current.get('symbol_role') == 'VALUE'))
        next_is_value = next_node['type'] in ('NUMBER', 'SYMBOL', 'GROUP')
        
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
def extract_variables(nodes, seen=None):
    """
    ASTを巡回し、予約語・組み込み定数以外の未知のシンボルを
    出現順に変数として抽出する（Auto-Detect機能）
    """
    if seen is None:
        seen = []
    for n in nodes:
        if n['type'] == 'SYMBOL':
            sym = n['value']
            if sym not in FUNCTION_SYMBOLS and sym not in BUILTIN_VALUE_SYMBOLS and sym not in seen:
                seen.append(sym)
        elif n['type'] == 'GROUP':
            extract_variables(n['children'], seen)
    return seen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typst Math to Python Transpiler")
    parser.add_argument("expr", type=str, help="Typstの数式表現 (例: '20 log( K / sqrt(1 + (2 pi f T)^2) )')")
    parser.add_argument("-v", "--vars", nargs="*", default=None, help="明示的に指定する変数リスト (例: -v f K T)")
    parser.add_argument("-e", "--eval", nargs="*", type=float, default=None, help="評価用の数値リスト (例: -e 1000 1.0 0.001)")
    
    args = parser.parse_args()
    
    try:
        # 1. パース (AST構築)
        ast = build_ast(args.expr)
        
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