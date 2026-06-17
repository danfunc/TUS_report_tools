import re
import numpy as np

class TypstMathTranspiler:
    def __init__(self):
        # Typstの数学関数・定数をNumPyのものにマッピング
        self.replacements = {
            r'\bpi\b': 'np.pi',
            r'\be\b': 'np.e',
            r'\bsin\b': 'np.sin',
            r'\bcos\b': 'np.cos',
            r'\btan\b': 'np.tan',
            r'\blog\b': 'np.log10', # ボード線図を想定して常用対数に
            r'\bln\b': 'np.log',
            r'\bsqrt\b': 'np.sqrt',
        }

    def transpile(self, typst_expr, variables):
        """
        Typstの数式文字列をPythonの実行可能な関数オブジェクトに変換する。
        
        :param typst_expr: Typstの数式文字列 (例: "K / (1 + s * T)")
        :param variables: 変数のリスト (例: ['s', 'K', 'T'])
        :return: 実行可能な関数 (lambda)
        """
        py_expr = typst_expr
        
        # 1. 累乗演算子の変換 (Typst: ^ -> Python: **)
        py_expr = py_expr.replace('^', '**')

        # 1.5 暗黙の乗算（スペース）の変換
        # 肯定先読み (?=...) を使用し、後ろの文字を消費せずに連続マッチさせる
        py_expr = re.sub(r'([0-9a-zA-Z)])\s+(?=[a-zA-Z(])', r'\1 * ', py_expr)
        
        # 2. 定数・関数の置換
        for pattern, py_func in self.replacements.items():
            py_expr = re.sub(pattern, py_func, py_expr)

            
        # 3. 暗黙の乗算の警告（完全なASTパースなしでは確実な変換が不可能な領域）
        # 例: 2s -> 2*s など。今回は簡易版のため明示的な * を要求する。
        if re.search(r'\d[a-zA-Z]', py_expr):
            print("[警告] 暗黙の乗算(例: 2x)が検出されました。トランスパイルが失敗する可能性があります。明示的に * を使用してください。")

        # 4. lambda式として動的コンパイル
        args_str = ", ".join(variables)
        lambda_str = f"lambda {args_str}: {py_expr}"

        print("トラスパイル後表現"+lambda_str)
        
        try:
            # 評価環境にnumpyを注入
            compiled_func = eval(lambda_str, {"np": np})
            print(f"[Transpile Success] {typst_expr}  =>  {lambda_str}")
            return compiled_func
        except SyntaxError as e:
            print(f"[Transpile Error] 構文エラー: {e}\n生成されたコード: {lambda_str}")
            return None

# ==========================================
# 実行テスト（ボード線図の理論式への組み込み）
# ==========================================
if __name__ == "__main__":
    transpiler = TypstMathTranspiler()
    
    # 1次遅れ系の伝達関数からゲイン(dB)を求めるTypstの数式
    # G(s) = K / (1 + sT), s = j * 2 * pi * f
    # 絶対値の20log10を取る数式を想定
    
    typst_formula = "20 log(K / sqrt(1+(2 pi f T)^2))"
    
    # トランスパイル (独立変数は f, パラメータは K, T)
    theory_func = transpiler.transpile(typst_formula, variables=['f', 'K', 'T'])
    
    if theory_func:
        # 理論カーブの描画用にテストデータを生成
        f_vals = np.logspace(1, 5, 100) # 10Hz ~ 100kHz
        
        # K=1 (0dB), T=1/(2*pi*1000) (折点周波数1kHz) として計算
        K_val = 1.0
        T_val = 1.0 / (2 * np.pi * 1000)
        
        # ベクトル演算として即座に実行可能
        gain_db = theory_func(f_vals, K_val, T_val)
        
        print(f"f=10Hz  における理論ゲイン: {gain_db[0]:.2f} dB")
        print(f"f=1kHz  における理論ゲイン: {gain_db[50]:.2f} dB")
        print(f"f=100kHzにおける理論ゲイン: {gain_db[-1]:.2f} dB")