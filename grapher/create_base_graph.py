import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.ticker as ticker

# グラフで使用するフォントを全体に設定
plt.rcParams['font.family'] = 'HackGen Console NF'

# グラフの保存先をカレントディレクトリに設定
plt.rcParams['savefig.directory'] = ''


def create_base_graph(xlabel="X-axis", ylabel="Y-axis", figsize=(8, 6)):
        # y軸ラベルからTeX形式の文字列を抽出
    tex_label_match = re.search(r'\$.*?\$', ylabel)
    tex_label = tex_label_match.group(0) if tex_label_match else "y"
    
    # x軸ラベルから変数名を抽出
    var_match = re.search(r'(\$.*?\$)', xlabel)
    var_name = var_match.group(1) if var_match else "x"
    """
    上下左右に内向きの目盛りがあり、左と下に数値とラベルが表示され、
    グリッドが非表示のグラフを作成します。

    Args:
        xlabel (str, optional): X軸のラベル。デフォルトは "X-axis"。
        ylabel (str, optional): Y軸のラベル。デフォルトは "Y-axis"。
        figsize (tuple, optional): グラフのサイズ。デフォルトは (8, 6)。

    Returns:
        tuple: matplotlib の Figure オブジェクトと Axes オブジェクトのタプル (fig, ax)。
    """
    # FigureとAxesオブジェクトを生成
    fig, ax = plt.subplots(figsize=figsize)

    # 軸ラベルを設定
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 副目盛りを有効にする
    ax.minorticks_on()

    # 主目盛りの設定
    ax.tick_params(
        which='major',
        direction='in',  # 目盛りを内向きに
        top=True,        # 上側の目盛りを表示
        right=True,      # 右側の目盛りを表示
        bottom=True,     # 下側の目盛りを表示
        left=True,       # 左側の目盛りを表示
        labeltop=False,  # 上側のラベル（数値）を非表示
        labelright=False,# 右側のラベル（数値）を非表示
        labelbottom=True,# 下側のラベル（数値）を表示
        labelleft=True   # 左側のラベル（数値）を表示
    )

    # 副目盛りの設定
    ax.tick_params(
        which='minor',
        direction='in',
        top=True,
        right=True,
        bottom=True,
        left=True
    )
    # グリッドを非表示に設定
    ax.grid(False)

    # x軸とy軸のラベルを10のべき乗形式で表示
    formatter = plt.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    return fig, ax


def add_equation_line(ax, func, label="Reference", color="gray", linestyle="--", **kwargs):
    """
    任意のグラフ（Axes）に、任意の関数（式）に基づく直線を挿入するユーティリティ関数。

    Args:
        ax (matplotlib.axes.Axes): 対象のAxesオブジェクト。
        func (callable): xを引数に取り、yを返す関数（例: lambda x: -x）。
        label (str, optional): 凡例に表示するラベル。デフォルトは "Reference"。
        color (str, optional): 線の色。デフォルトは "gray"。
        linestyle (str, optional): 線のスタイル。デフォルトは "--"。
        **kwargs: ax.plotに渡す追加のキーワード引数。
    """
    x_lim = ax.get_xlim()
    x_vals = np.linspace(x_lim[0], x_lim[1], 100)
    y_vals = func(x_vals)
    ax.plot(x_vals, y_vals, label=label, color=color, linestyle=linestyle, **kwargs)
    # 描画範囲を維持する
    ax.set_xlim(x_lim)


def plot_regression(x, y, xlabel, ylabel, data_label="測定データ1", filename=None, x2=None, y2=None, data_label2="測定データ2", save=True):
    """
    データプロットと回帰直線をグラフに描画し、統計情報を表示します。
    2系列のデータを処理できます。

    Args:
        x (pd.Series or np.array): 1系列目のx軸データ。
        y (pd.Series or np.array): 1系列目のy軸データ。
        xlabel (str): x軸のラベル。
        ylabel (str): y軸のラベル。
        data_label (str, optional): 1系列目の測定データの凡例ラベル。
        filename (str, optional): グラフを保存するファイル名。
        x2 (pd.Series or np.array, optional): 2系列目のx軸データ。
        y2 (pd.Series or np.array, optional): 2系列目のy軸データ。
        data_label2 (str, optional): 2系列目の測定データの凡例ラベル。
    """
    # グラフのベースを作成
    fig, ax = create_base_graph(xlabel=xlabel, ylabel=ylabel)

    # y軸ラベルからTeX形式の文字列を抽出
    tex_label_match = re.search(r'\$.*?\$', ylabel)
    tex_label = tex_label_match.group(0) if tex_label_match else "y"
    
    # x軸ラベルから変数名を抽出
    var_match = re.search(r'(\$.*?\$)', xlabel)
    var_name = var_match.group(1) if var_match else "x"

    def fmt(val):
        s = f"{val:.2e}"
        base, exponent = s.split('e')
        return f"${base} \\times 10^{{{int(exponent)}}}$"

    # 1系列目の散布図と回帰
    ax.scatter(x, y, label=data_label, color='red')
    p, V = np.polyfit(x, y, 1, cov=True)
    A, a = p
    SE_A, SE_a = np.sqrt(np.diag(V))
    
    legend_label = f'回帰直線1: {tex_label} = ({fmt(A)} ± {fmt(SE_A)}){var_name} + ({fmt(a)} ± {fmt(SE_a)})'
    
    y_pred = A * x + a
    residuals = y - y_pred
    SSR = np.sum(residuals**2)
    df = len(x) - 2
    ser = np.sqrt(SSR / df) if df > 0 else 0

    print("--- 1系列目の回帰分析結果 ---")
    print(f"傾き (A): {A:.10f} (標準誤差: {SE_A:.10f})")
    print(f"切片 (a): {a:.10f} (標準誤差: {SE_a:.10f})")
    print(f"回帰の標準誤差 (SER): {ser:.10f}")
    print("--------------------")

    # 2系列目の処理
    if x2 is not None and y2 is not None:
        ax.scatter(x2, y2, label=data_label2, color='blue', marker='x')
        p2, V2 = np.polyfit(x2, y2, 1, cov=True)
        A2, a2 = p2
        SE_A2, SE_a2 = np.sqrt(np.diag(V2))
        
        legend_label2 = f'回帰直線2: {tex_label} = ({fmt(A2)} ± {fmt(SE_A2)}){var_name} + ({fmt(a2)} ± {fmt(SE_a2)})'

        y_pred2 = A2 * x2 + a2
        residuals2 = y2 - y_pred2
        SSR2 = np.sum(residuals2**2)
        df2 = len(x2) - 2
        ser2 = np.sqrt(SSR2 / df2) if df2 > 0 else 0

        print("--- 2系列目の回帰分析結果 ---")
        print(f"傾き (A): {A2:.10f} (標準誤差: {SE_A2:.10f})")
        print(f"切片 (a): {a2:.10f} (標準誤差: {SE_a2:.10f})")
        print(f"回帰の標準誤差 (SER): {ser2:.10f}")
        print("--------------------")

    # 軸の範囲を確定し、自動スケーリングを無効化
    ax.autoscale_view()
    ax.set_autoscale_on(False)

    # 描画範囲いっぱいに回帰直線を引く
    x_lim = ax.get_xlim()
    x_fit = np.linspace(x_lim[0], x_lim[1], 100)
    
    # 1系列目の回帰直線
    y_fit = A * x_fit + a
    ax.plot(x_fit, y_fit, color='red', linestyle='-', label=legend_label)

    # 2系列目の回帰直線
    if 'p2' in locals():
        y_fit2 = A2 * x_fit + a2
        ax.plot(x_fit, y_fit2, color='blue', linestyle='--', label=legend_label2)

    # 凡例を表示
    ax.legend()

    # グラフを表示または保存
    if save:
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

    return fig, ax
