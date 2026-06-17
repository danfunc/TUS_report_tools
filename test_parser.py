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
# 実行テスト
# ==========================================
if __name__ == "__main__":
    transpiler = TypstMathTranspiler()
    
    # テスト式: お前が躓いた暗黙の乗算、未定義シンボル検証、関数呼び出しが全て含まれる
    typst_formula = "20 log( K / sqrt(1 + (2 pi f T)^2) )"
    typst_formula2 = "(R_2 sqrt(1+omega^2 C_1^2 R_1^2))/sqrt((R_1+R_2+R_3-omega^2 L_1 C_1 R_1)^2+omega^2{C_1 R_1(R_2+R_3)+L_1}^2)"
    
    # 変数 f, K, T を登録してコンパイル
    #func = transpiler.transpile(typst_formula, variables=["f","K","T"])
    func2 = transpiler.transpile(typst_formula2, variables=["R_1","R_2","R_3","C_1","L_1","omega"])
    func = 1

        