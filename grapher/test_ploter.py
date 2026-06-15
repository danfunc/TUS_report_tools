import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'HackGen Console NF'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

def extract_best_asymptotes(x, y, expected_slopes=[0, -20], tol=7.0, min_points=3, r_squared_threshold=0.70):
    """
    複数の漸近線を抽出し、同一実験での一貫性を保つ
    各グラフごとに複数本の直線を検出
    """
    log_x = np.log10(x)
    
    lines = []
    used_indices = set()
    
    for target_slope in expected_slopes:
        best_line = None
        best_score = -float('inf')
        best_indices = set()
        
        # 目標勾配に近い点のみを候補に
        gradients_approx = np.gradient(y, log_x)
        candidate_mask = np.abs(gradients_approx - target_slope) < tol
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) < min_points:
            print(f"[棄却] 目標 {target_slope:6.1f} dB/dec -> 候補点不足（{len(candidate_indices)}/{min_points}）")
            continue
        
        # スライディングウィンドウで連続領域を探索
        for start_idx in range(len(candidate_indices) - min_points + 1):
            for end_idx in range(start_idx + min_points, len(candidate_indices) + 1):
                window_indices = candidate_indices[start_idx:end_idx]
                
                # 既に採用されたインデックスとの重複チェック（70%以上はスキップ）
                overlap = len(set(window_indices) & used_indices) / len(window_indices)
                if overlap > 0.7:
                    continue
                
                chunk_log_x = log_x[window_indices]
                chunk_y = y[window_indices]
                
                # 直線フィッティング
                res = linregress(chunk_log_x, chunk_y)
                y_pred = res.slope * chunk_log_x + res.intercept
                
                # R^2 を計算
                ss_res = np.sum((chunk_y - y_pred)**2)
                ss_tot = np.sum((chunk_y - np.mean(chunk_y))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -float('inf')
                
                # 目標勾配への誤差
                slope_error = abs(res.slope - target_slope)
                
                # スコア関数：目標勾配への接近度を最優先
                score = -slope_error + 0.1 * (r_squared - r_squared_threshold)
                
                # 採用条件
                if slope_error < tol and r_squared > r_squared_threshold and score > best_score:
                    best_score = score
                    best_line = {
                        'target': target_slope,
                        'A': res.slope,
                        'a': res.intercept,
                        'r_squared': r_squared,
                        'slope_error': slope_error,
                        'x_pts': chunk_log_x,
                        'y_pts': chunk_y
                    }
                    best_indices = set(window_indices)
        
        if best_line is not None:
            lines.append(best_line)
            used_indices.update(best_indices)
            print(f"[採用✓] 目標 {best_line['target']:6.1f} dB/dec -> 実測勾配: {best_line['A']:6.2f} dB/dec (誤差: {best_line['slope_error']:5.2f}, R²: {best_line['r_squared']:.3f}, 点数: {len(best_line['x_pts'])})")
        else:
            print(f"[棄却] 目標 {target_slope:6.1f} dB/dec -> 適切な直線なし")

    # 交点（折点周波数）の計算
    intersections = []
    if len(lines) >= 2:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                A1, a1 = lines[i]['A'], lines[i]['a']
                A2, a2 = lines[j]['A'], lines[j]['a']
                
                if np.abs(A1 - A2) < 0.5:
                    continue
                    
                cross_log_x = (a2 - a1) / (A1 - A2)
                cross_f = 10 ** cross_log_x
                cross_y = A1 * cross_log_x + a1
                intersections.append((cross_f, cross_y))
                print(f"[交点] 周波数: {cross_f:.2f} Hz (ゲイン: {cross_y:.2f} dB)")
    
    return lines, intersections

# 描画部分は前回の plot_data_driven_bode と同様のため省略。
# 呼び出し元の expected_slopes には回路特性に合わせたリストを渡せ。

def plot_data_driven_bode(sheet_name):
    # データ読み込み
    try:
        sheet = pd.read_excel("結果.xlsx", sheet_name=sheet_name)
    except FileNotFoundError:
        print("エラー: 結果.xlsx が見つからない。")
        return

    x = sheet["f"].values
    y = 20 * np.log10(sheet["Vout"] / sheet["Vin"]).values

    # 回路が持つと予想される漸近線の傾きを指定しろ（1次遅れなら 0, -20。2次遅れが見込まれるなら -40 を追加）
    # tol はデータに含まれるノイズの大きさに合わせて調整する
    lines, intersections = extract_best_asymptotes(x, y, expected_slopes=[15, 0, -20, -40], tol=10.0, min_points=2, r_squared_threshold=0.60)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xscale("log", base=10)
    ax.set_xlabel(r"周波数 [Hz]", fontsize=13, fontweight='bold')
    ax.set_ylabel(r"電圧利得 [dB]", fontsize=13, fontweight='bold')
    
    # グリッド設定を改善
    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.4, color='gray')
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2, color='gray')
    ax.set_axisbelow(True)

    # 生データのプロット
    ax.plot(x, y, color="gray", linestyle="-", marker="o", markersize=7, alpha=0.6, 
            linewidth=1.8, label="測定値", zorder=1)
    
    # 軸の範囲を確定
    y_vals = y
    for line in lines:
        y_vals = np.concatenate([y_vals, line['y_pts']])
    
    margin = (np.max(y_vals) - np.min(y_vals)) * 0.1
    ax.set_ylim(np.min(y_vals) - margin, np.max(y_vals) + margin)

    # 抽出された漸近線のプロット
    x_plot = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, line_data in enumerate(lines):
        color = colors[idx % len(colors)]
        
        # フィッティングに採用されたデータ点を強調表示
        ax.scatter(10**line_data['x_pts'], line_data['y_pts'], 
                  color=color, s=70, zorder=3, alpha=0.85, marker='o',
                  edgecolors='darkgray', linewidth=0.7,
                  label=f"採用データ (目標:{line_data['target']}dB/dec)")
        
        # 漸近線の描画
        y_plot = line_data['A'] * np.log10(x_plot) + line_data['a']
        ax.plot(x_plot, y_plot, color=color, linestyle="--", linewidth=2.8, zorder=2, 
               label=f"漸近線 (実測勾配:{line_data['A']:.2f}dB/dec)")

    # 交点（折点周波数）のプロット
    for idx, (cross_f, cross_y) in enumerate(intersections):
        # グラフの描画範囲内にある交点のみ描画
        if np.min(x) * 0.05 <= cross_f <= np.max(x) * 20:
            ax.scatter(cross_f, cross_y, color="black", marker="X", s=180, zorder=5, 
                      edgecolors='white', linewidth=1.8,
                      label=f"折点周波数: {cross_f:.2f}Hz")

    # 軸のフォーマット改善
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}' if x < 1000 else f'{x/1000:.1f}k'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    
    # 凡例を最適化
    ax.legend(loc="best", fontsize=10, framealpha=0.97, 
             edgecolor='darkgray', fancybox=True, shadow=True)
    ax.tick_params(which='both', direction='in', labelsize=11)
    
    # 背景色を微調整
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    fig.savefig(f"{sheet_name}_data_driven.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ {sheet_name}_data_driven.png を保存しました")
    plt.close(fig)

def main():
    plot_data_driven_bode("TA(2)100mV")
    plot_data_driven_bode("TA(2)500mV")

if __name__ == "__main__":
    main()