import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pwlf

# 既存のグラフ作成関数をインポート（環境に合わせて調整しろ）
from grapher.create_base_graph import create_base_graph

def plot_bode_piecewise(sheet_name, n_segments=3):
    """
    区分線形回帰を用いて、ボード線図の実測データから
    連続する漸近線（折れ線）と折点周波数を抽出・描画する。
    """
    # データの読み込み
    sheet = pd.read_excel("結果.xlsx", sheet_name=sheet_name)
    x = sheet["f"].values
    y = 20 * np.log10(sheet["Vout"] / sheet["Vin"]).values

    # ボード線図の横軸は対数スケールであるため、フィッティングも対数空間で行う
    log_x = np.log10(x)

    # 区分線形回帰モデルの初期化とフィッティング
    my_pwlf = pwlf.PiecewiseLinFit(log_x, y)
    
    # n_segments個の直線でフィッティングし、最適な折点(breaks)を自動探索
    # ※ 1次遅れなら2セグメント(平坦域と減衰域)、帯域通過なら3セグメント等、回路の理論に合わせろ
    breaks = my_pwlf.fit(n_segments)

    # 予測用データの生成
    x_pred = np.linspace(min(log_x), max(log_x), 100)
    y_pred = my_pwlf.predict(x_pred)

    # --- 描画処理 ---
    fig, ax = create_base_graph(xlabel=r"周波数[Hz]", ylabel=r"電圧利得[dB]")
    ax.set_xscale("log", base=10)
    
    # 実測値のプロット
    ax.scatter(x, y, label="測定値", color="blue", marker="o", alpha=0.4)

    # 抽出された漸近線（折れ線）のプロット
    ax.plot(10**x_pred, y_pred, color="red", linestyle="-", linewidth=2.5, 
            label=f"区分線形近似 ({n_segments} segments)")

    # 推定された折点周波数のプロット（物理的な極・零点の候補位置）
    ax.scatter(10**breaks, my_pwlf.predict(breaks), color="black", marker="x", 
               s=100, zorder=5, label="推定折点周波数")

    ax.legend()
    fig.savefig(f"{sheet_name}_pwlf.png")
    plt.close(fig)

    # --- 物理特性の出力 ---
    print(f"=== {sheet_name} の解析結果 ===")
    slopes = my_pwlf.slopes
    for i in range(n_segments):
        # 傾きの単位はそのまま dB/dec となる
        print(f"セグメント {i+1} の傾き: {slopes[i]:.2f} dB/dec")
        
        if i < n_segments - 1:
            # 折点周波数（対数空間からHzに戻す）
            fc = 10**breaks[i+1]
            print(f"  -> 折点周波数 (極/零点候補): {fc:.2f} Hz")
    print("==============================\n")

def main():
    # 対象の回路モデル（1次遅れ、2次遅れ等）に応じてセグメント数を指定しろ
    # 例: ゲインが平坦な領域と、高域で減衰する領域の2つなら segments=2
    plot_bode_piecewise("TA(2)100mV", n_segments=3)
    plot_bode_piecewise("TA(2)500mV", n_segments=3)
    

if __name__ == "__main__":
    main()
