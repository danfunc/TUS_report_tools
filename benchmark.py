import timeit
import numpy as np
from test_parser import *

# 前回の TypstMathTranspiler クラスがここにある前提

def run_benchmark():
    transpiler = TypstMathTranspiler()

    # ==========================================
    # ストレステストの定義
    # ==========================================
    test_cases = {
        # 1. ベースライン（最小構成）
        # 目的: 関数の呼び出しオーバーヘッドや基本パスの最低遅延を測る
        "Baseline": {
            "expr": "a + b",
            "vars": ['a', 'b']
        },
        
        # 2. フラット＆ブロード（字句解析・意味解析の走査負荷）
        # 目的: 階層は浅いがトークン数が異常に多い場合。正規表現エンジンの負荷と、
        #       暗黙の乗算パス（隣接ノードの総当たりチェック）のO(N)負荷を検証する。
        "Flat & Broad": {
            "expr": "a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p",
            "vars": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
        },

        # 3. ディープネスト（構文解析の再帰・スタック負荷）
        # 目的: 異常な深さの括弧。build_ast関数とgenerate_code関数の再帰呼び出し上限や、
        #       Pythonのコールスタックに対するストレステスト。
        "Deep Nesting": {
            "expr": "log(sin(cos(tan(sqrt(ln(log(sin(x))))))))",
            "vars": ['x']
        },

        # 4. 暗黙の乗算の極致（意味解析の分岐負荷）
        # 目的: 全てがスペース区切り。Lexerが大量のSKIPを処理し、Semantic Analyzerが
        #       全てのトークン間に '*' をねじ込む最悪ケースの負荷を測る。
        "Heavy Implicit": {
            "expr": "2 pi f L C R x y z 100",
            "vars": ['f', 'L', 'C', 'R', 'x', 'y', 'z']
        },

        # 5. 現実的な高負荷（実践投入レベル）
        # 目的: 2次ローパスフィルタの伝達関数ゲイン。分数、累乗、関数、暗黙の乗算が
        #       全て入り混じった、レポートで実際に使うレベルの総合負荷。
        "Realistic LPF": {
            "expr": "20 log( 1 / sqrt( (1 - (f/f_c)^2)^2 + (2 zeta f / f_c)^2 ) )",
            "vars": ['f', 'f_c', 'zeta']
        }
    }

    print(f"{'Test Case':<15} | {'Compiles/sec':<15} | {'Time per Compile (μs)':<20}")
    print("-" * 55)

    # 各テストケースを10,000回コンパイルして平均を出す
    iterations = 10000

    for name, data in test_cases.items():
        expr = data['expr']
        variables = data['vars']
        
        # 動的コンパイル（eval）まで含めるとPython側のオーバーヘッドが大きすぎるため、
        # 今回は純粋に「トランスパイラとしての文字列生成」までの速度を計測する
        # ※ evalまで測りたい場合は transpiler.transpile() を直接呼べ
        setup_code = f"""
from __main__ import TypstMathTranspiler, build_ast, analyze_semantics, generate_code
transpiler = TypstMathTranspiler()
expr = "{expr}"
variables = {variables}
"""
        test_code = """
ast = build_ast(expr)
ast_semantic = analyze_semantics(ast, variables)
py_code_body = generate_code(ast_semantic)
"""
        
        try:
            # 実行時間の計測
            total_time = timeit.timeit(stmt=test_code, setup=setup_code, number=iterations)
            
            time_per_compile = (total_time / iterations) * 1_000_000 # マイクロ秒に変換
            compiles_per_sec = iterations / total_time
            
            print(f"{name:<15} | {compiles_per_sec:,.0f} 次/秒   | {time_per_compile:.2f} μs")
            
        except Exception as e:
            print(f"{name:<15} | ERROR: {e}")

if __name__ == "__main__":
    run_benchmark()