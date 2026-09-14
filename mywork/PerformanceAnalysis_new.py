"""教学版性能分析脚本——只保留「加密/解密对比」和「直方图分析」。

直接运行本文件即可：
    python mywork/PerformanceAnalysis_new.py

参数都在文件顶部 CONFIG 区，一键修改。
所有加密/解密调用都平铺直叙，没有中间层。
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pil_image
from sympy import im

from .Encryption import Encrypter, DeEncrypter, EncryptionConfig

# ============================================================
# CONFIG：所有可调参数都在这一坨
# ============================================================

# 加密配置（保持和之前一致）
ENCRYPTION_CONFIG = EncryptionConfig(
    seed=2026,
    b_max=64,
    b_min=4,
    block_operation="xor",
    global_parallel_size=2,
    global_permutation_method="global_bit_permutation_v2",
    cross_plane_diffusion = True,
    cross_plane_shift = 1,
    bidirectional_diffusion = False,
)

# 测试图片列表
IMAGE_PATHS = [
    r"C:\ImageEncryption\images\img1.png",
    r"C:\ImageEncryption\images\img2.png",
    r"C:\ImageEncryption\images\img3.png",
    r"C:\ImageEncryption\images\img4.png",
    r"C:\ImageEncryption\images\img5.png",
    # r"C:\ImageEncryption\images\img6.png",  # 需要时解开注释
]

# 直方图输出目录（histogram_grid.png 会存在这里）
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

# 直方图样式参数
HIST_DPI = 100                     # 保存图片的 DPI
HIST_FONT_FAMILY = "Times New Roman"
HIST_BAR_ALPHA = 0.6               # RGB 三通道直方图重叠时的透明度
HIST_USE_REFERENCE_STYLE = True    # 使用带网格的论文风格

# 直方图多轮密钥轮次：设为 5 表示每张图用 5 个不同 user_key 各加密一次，HV 取平均
NUM_USER_KEY_ROUNDS = 1

# ============================================================
# 初始化加密器/解密器（全局唯一，直接用）
# ============================================================

MyEncrypter = Encrypter(ENCRYPTION_CONFIG)
MyDeEncrypter = DeEncrypter(ENCRYPTION_CONFIG)

# ============================================================
# 通用小工具
# ============================================================

def print_table(headers, rows):
    """打印一个 ASCII 表格，第一列左对齐，其余右对齐。"""
    if not headers:
        return
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(headers[i])), max(len(row[i]) for row in text_rows))
        for i in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def format_row(values, is_header=False):
        cells = [
            value.center(widths[i]) if is_header
            else value.ljust(widths[i]) if i == 0
            else value.rjust(widths[i])
            for i, value in enumerate(values)
        ]
        return "| " + " | ".join(cells) + " |"

    print(separator)
    print(format_row(headers, is_header=True))
    print(separator)
    for row in text_rows:
        print(format_row(row))
    print(separator)


def load_rgb_image(image_input):
    """把路径/数组统一转成 RGB uint8 数组。"""
    if isinstance(image_input, (str, Path)):
        with pil_image.open(image_input) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image_input)
    if arr.ndim == 2:
        return np.asarray(pil_image.fromarray(arr).convert("RGB"), dtype=np.uint8)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2).astype(np.uint8)
    return arr[..., :3].astype(np.uint8)


def histogram_variance(hist):
    """HV = 1/256 * sum_k (h_k - mean(h))^2，越小越均匀。"""
    hist = np.asarray(hist, dtype=np.float64)
    return float(np.mean((hist - hist.mean()) ** 2))


def number_to_letters(n):
    """0->(a), 25->(z), 26->(aa)，用于给每个小图标号。"""
    letters = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        letters = chr(ord("a") + r) + letters
    return f"({letters})"


# ============================================================
# 测试 1：加密 / 解密 / 原图 对比 + 计时
# ============================================================

def run_encryption_decryption_test(show=True, save_path=None, dpi=150):
    """对每张图跑一遍加密+解密，画 3 列对比图并打印计时表。

    参数
    ----
    show : 是否 plt.show() 弹窗显示
    save_path : 保存对比图的路径（None 则不保存）
    dpi : 保存图片的 DPI
    """
    rows = []        # 打印表格用的数据
    stored_images = []  # (原图, 密文图, 解密图)，用于最后统一画图

    for image_path in IMAGE_PATHS:
        # ---------- 加密 ----------
        # MyEncrypter.set_cml(image=image_path)   # 初始化图像依赖的混沌系统
        MyEncrypter.set_user_key(image=image_path)  # 初始化图像依赖的 user_key
        start_time = time.perf_counter()
        encryption_result = MyEncrypter.encrypt(image_path, print_profile=True)
        encrypt_seconds = time.perf_counter() - start_time

        encrypted_image = encryption_result.encrypted_image
        metadata = encryption_result.metadata
        MyEncrypter.resume_cml()                # 清空状态，下一张图重新初始化

        # ---------- 解密 ----------
        start_time = time.perf_counter()
        decryption_result = MyDeEncrypter.decrypt(
            encrypted_image, metadata, print_profile=False
        )
        decrypt_seconds = time.perf_counter() - start_time
        decrypted_image = decryption_result.decrypted_image

        # ---------- 检查是否完全恢复 ----------
        original_image = load_rgb_image(image_path)
        exact_recovery = np.array_equal(original_image, decrypted_image)

        rows.append({
            "name": Path(image_path).name,
            "encrypt_seconds": encrypt_seconds,
            "decrypt_seconds": decrypt_seconds,
            "exact_recovery": exact_recovery,
        })
        stored_images.append((original_image, encrypted_image, decrypted_image))

    # ---------- 打印计时表格 ----------
    print("\nEncryption/decryption image comparison")
    print("--------------------------------------")
    table_rows = [
        (
            row["name"],
            f"{row['encrypt_seconds']:.6f}",
            f"{row['decrypt_seconds']:.6f}",
            "Yes" if row["exact_recovery"] else "No",
        )
        for row in rows
    ]
    print_table(
        ("Image", "Encrypt (s)", "Decrypt (s)", "Exact Recovery"),
        table_rows,
    )

    # ---------- 画 3 列对比图 ----------
    num_images = len(IMAGE_PATHS)
    fig, axes = plt.subplots(
        num_images, 3,
        figsize=(12, 3.8 * num_images),
        squeeze=False,
        constrained_layout=True,
    )
    column_titles = ("Original", "Encrypted", "Decrypted")

    for row_idx, (original, encrypted, decrypted) in enumerate(stored_images):
        for col_idx, img in enumerate((original, encrypted, decrypted)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=13, fontweight="bold")
            ax.axis("off")
        # 在最左边竖着写图片名
        axes[row_idx, 0].text(
            -0.04, 0.5, rows[row_idx]["name"],
            transform=axes[row_idx, 0].transAxes,
            rotation=90, ha="right", va="center", fontsize=10,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"saved figure: {save_path}")

    if show:
        plt.show()

    return rows, fig


# ============================================================
# 测试 2：直方图 + 直方图方差 HV
# ============================================================

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_histogram_test(
    show=True,
    save_dir=None,
    show_sublabels=False,
    save_subfigures=False,
):
    """每张图循环 NUM_USER_KEY_ROUNDS 次不同 user_key 加密，HV 取均值后画图+做表。

    彩色图统一 4 列：原图 | RGB 原图直方图 | 密文（第 0 轮） | RGB 密文直方图（第 0 轮）。
    灰度图会先把密文转成灰度再统计，画图也按 RGB 处理。
    表格里每个通道的 HV 是该图在所有轮次上的均值；最后一行是多张图的通道均值+总体均值。

    参数:
        show: 是否调用 plt.show() 展示图形。
        save_dir: 拼接总大图保存目录，为 None 时不保存总图。
        show_sublabels: 控制每个子图下方 (a), (b), ... 标签是否显示，默认 False（不开启）。
        save_subfigures: 是否在 'mywork/figure/' 下新建独立目录并逐一导出单个子图，默认 False（不开启）。
    """
    num_rounds = NUM_USER_KEY_ROUNDS
    if num_rounds <= 0:
        raise ValueError("NUM_USER_KEY_ROUNDS must be a positive integer")

    rgb_colors = {"R": "#d84a4a", "G": "#3f9e52", "B": "#416da5"}
    gray_colors = {"Gray": "#9c9c9c"}
    encrypted_rgb_colors = {"R": "#932525", "G": "#21612f", "B": "#1f3153"}
    encrypted_gray_colors = {"Gray": "#4d4d4d"}
    pixel_ticks = [0, 45, 90, 135, 180, 225, 255]

    run_data = []

    for image_path in IMAGE_PATHS:
        # ---------- 一次性读原图、算原图直方图 ----------
        original_image = load_rgb_image(image_path)
        num_channels = original_image.shape[2]
        channel_names = ["R", "G", "B"] if num_channels == 3 else ["Gray"]

        original_hists = {}
        if num_channels == 3:
            for i, name in enumerate(["R", "G", "B"]):
                original_hists[name] = np.bincount(
                    original_image[..., i].ravel(), minlength=256
                )
        else:
            gray = np.mean(original_image, axis=2).astype(np.uint8)
            original_hists["Gray"] = np.bincount(gray.ravel(), minlength=256)

        # ---------- 多轮不同 user_key 加密，统计 HV ----------
        hv_per_round = []
        exact_per_round = []
        first_round_image = None
        first_round_hists = None
        num_rounds = 5
        for round_idx in range(num_rounds):
            MyEncrypter.set_user_key(f"HISTTEST-{round_idx}", image=image_path)
            if num_rounds == 1:
                MyEncrypter.set_user_key(image=image_path)
            #使用这一组参数效果更是逆天好。
            # MyEncrypter.set_cml(mu=5,v=20,alpha=5,beta=5)
            encryption_result = MyEncrypter.encrypt(image_path, print_profile=False)
            encrypted_image = encryption_result.encrypted_image
            metadata = encryption_result.metadata

            decryption_result = MyDeEncrypter.decrypt(
                encrypted_image, metadata, print_profile=False
            )
            decrypted_image = decryption_result.decrypted_image
            exact_per_round.append(np.array_equal(original_image, decrypted_image))
            MyEncrypter.resume_user_key()

            # 计算当轮密文的直方图和 HV
            current_hists = {}
            current_hv = {}
            if num_channels == 3:
                for i, name in enumerate(["R", "G", "B"]):
                    hist = np.bincount(encrypted_image[..., i].ravel(), minlength=256)
                    current_hists[name] = hist
                    current_hv[name] = histogram_variance(hist)
            else:
                gray_pixels = np.mean(encrypted_image, axis=2).astype(np.uint8)
                hist = np.bincount(gray_pixels.ravel(), minlength=256)
                current_hists["Gray"] = hist
                current_hv["Gray"] = histogram_variance(hist)
            hv_per_round.append(current_hv)

            if round_idx == 0:
                first_round_image = encrypted_image
                first_round_hists = current_hists

        mean_hv = {}
        for ch in channel_names:
            values = [one_round[ch] for one_round in hv_per_round]
            mean_hv[ch] = float(np.mean(values))
        mean_hv["average"] = float(
            np.mean([one_round[ch] for one_round in hv_per_round for ch in channel_names])
        )

        run_data.append({
            "name": Path(image_path).name,
            "original_image": original_image,
            "original_hists": original_hists,
            "encrypted_image": first_round_image,
            "encrypted_hists": first_round_hists,
            "encrypted_hv": mean_hv,
            "channel_names": channel_names,
            "exact_recovery": all(exact_per_round),
        })

    # ---------- 打印 HV 表格 ----------
    print("\nHistogram analysis")
    print("------------------")
    channel_order = [c for c in ["R", "G", "B", "Gray"]
                     if any(c in d["encrypted_hv"] for d in run_data)]

    table_rows = []
    for data in run_data:
        row = [data["name"]]
        for ch in channel_order:
            value = data["encrypted_hv"].get(ch)
            row.append(f"{value:.2f}" if value is not None else "-")
        row.append(f"{data['encrypted_hv']['average']:.2f}")
        row.append("Yes" if data["exact_recovery"] else "No")
        table_rows.append(row)

    mean_row = ["Mean"]
    all_channel_values = []
    for ch in channel_order:
        values = [d["encrypted_hv"][ch] for d in run_data if ch in d["encrypted_hv"]]
        mean_row.append(f"{np.mean(values):.2f}" if values else "-")
        all_channel_values.extend(values)
    mean_row.append(f"{np.mean(all_channel_values):.2f}")
    mean_row.append("Yes" if all(d["exact_recovery"] for d in run_data) else "No")
    table_rows.append(mean_row)

    headers = ("Image",) + tuple(f"{ch} HV" for ch in channel_order) \
              + ("Average HV", "Exact Recovery")
    print_table(headers, table_rows)

    # ---------- 画直方图 ----------
    max_columns = 4
    num_rows = len(run_data)
    rect_bottom = 0.035 if show_sublabels else 0.0
    fig, axes = plt.subplots(
        num_rows, max_columns,
        figsize=(2.55 * max_columns, 2.6 * num_rows),
        squeeze=False,
        constrained_layout={"rect": (0.0, rect_bottom, 1.0, 1.0)},
    )

    if HIST_USE_REFERENCE_STYLE:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": [HIST_FONT_FAMILY, "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "axes.labelsize": 11,
            "grid.color": "#d4d4d4",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
        })

    panel_index = 0
    for row_idx, data in enumerate(run_data):
        # 1. 原图
        ax = axes[row_idx, 0]
        ax.imshow(data["original_image"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax._caption_text = number_to_letters(panel_index); panel_index += 1

        # 2. 原图直方图
        ax = axes[row_idx, 1]
        palette = rgb_colors if len(data["channel_names"]) == 3 else gray_colors
        for name, hist in data["original_hists"].items():
            ax.hist(np.arange(256), bins=np.arange(257), weights=hist,
                    color=palette[name], alpha=HIST_BAR_ALPHA, linewidth=0)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_xticks(pixel_ticks)
        ax.tick_params(axis="both", labelsize=7, pad=1)
        ax.set_axisbelow(True)
        ax._caption_text = number_to_letters(panel_index); panel_index += 1

        # 3. 密文图
        ax = axes[row_idx, 2]
        ax.imshow(data["encrypted_image"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax._caption_text = number_to_letters(panel_index); panel_index += 1

        # 4. 密文直方图
        ax = axes[row_idx, 3]
        palette = encrypted_rgb_colors if len(data["channel_names"]) == 3 else encrypted_gray_colors
        for name, hist in data["encrypted_hists"].items():
            ax.hist(np.arange(256), bins=np.arange(257), weights=hist,
                    color=palette[name], alpha=HIST_BAR_ALPHA, linewidth=0)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_xticks(pixel_ticks)
        ax.tick_params(axis="both", labelsize=7, pad=1)
        ax.set_axisbelow(True)
        ax._caption_text = number_to_letters(panel_index); panel_index += 1

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ---------- 按需绘制 (a), (b)... 子图标号 ----------
    if show_sublabels:
        for row_axes in axes:
            bottoms = []
            for ax in row_axes:
                if hasattr(ax, "_caption_text"):
                    bottoms.append(ax.get_window_extent(renderer).y0)
            if not bottoms:
                continue
            baseline = min(bottoms) - 6.0 * fig.dpi / 72.0
            for ax in row_axes:
                text = getattr(ax, "_caption_text", None)
                if text is None:
                    continue
                bbox = ax.get_window_extent(renderer)
                fig.text(
                    (bbox.x0 + bbox.x1) / 2.0 / fig.bbox.width,
                    baseline / fig.bbox.height,
                    text,
                    ha="center", va="top", fontsize=11,
                )

    # ---------- 逐个子图单独保存至 mywork/figure/<timestamp> ----------
    if save_subfigures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path("mywork/figures") / f"single_plots_{timestamp}"
        target_dir.mkdir(parents=True, exist_ok=True)

        col_suffixes = ["orig", "orig_hist", "cipher", "cipher_hist"]
        for row_idx, data in enumerate(run_data):
            stem = Path(data["name"]).stem
            for col_idx in range(max_columns):
                ax = axes[row_idx, col_idx]
                bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                out_path = target_dir / f"{stem}_{col_suffixes[col_idx]}.png"
                fig.savefig(out_path, bbox_inches=bbox, dpi=HIST_DPI, facecolor="white")
        print(f"saved individual subfigures to: {target_dir}")

    # ---------- 保存整张拼接图 ----------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "histogram_grid.png",
                    dpi=HIST_DPI, facecolor="white")
        print(f"saved figure: {save_dir / 'histogram_grid.png'}")

    if show:
        plt.show()

    return run_data, fig


try:
    from scipy.stats import chi2 as chi2_dist
except ImportError:
    chi2_dist = None

# ============================================================
# 测试 3：卡方检验（Chi-Square Test）
# ============================================================

def calculate_chi_square(hist: np.ndarray, total_pixels: int) -> float:
    """计算 256 阶灰度级直方图与均匀分布的卡方值 (df=255)。"""
    expected = total_pixels / 256.0
    return float(np.sum((hist - expected) ** 2 / expected))


def run_chi_square_test(alpha=0.05):
    """每张图循环 NUM_USER_KEY_ROUNDS 次不同 user_key 加密，计算密文卡方值并制表。

    统计每个通道的卡方检验值并在多轮上取均值，判定是否通过卡方检验（均值 < 临界值）。
    表格包含各通道卡方值、平均卡方值以及检验结果；最后一行给出全部图片的统计均值。
    """
    num_rounds = NUM_USER_KEY_ROUNDS
    if num_rounds <= 0:
        raise ValueError("NUM_USER_KEY_ROUNDS must be a positive integer")

    # 计算显著性水平 alpha 下自由度 df=255 的卡方临界值 (α=0.05 时约 293.25)
    if chi2_dist is not None:
        critical_value = float(chi2_dist.ppf(1.0 - alpha, df=255))
    else:
        critical_value = 293.2478 if alpha == 0.05 else (310.4574 if alpha == 0.01 else 293.2478)

    run_data = []

    for image_path in IMAGE_PATHS:
        original_image = load_rgb_image(image_path)
        num_channels = original_image.shape[2]
        channel_names = ["R", "G", "B"] if num_channels == 3 else ["Gray"]

        chi2_per_round = []

        num_rounds = 5
        for round_idx in range(num_rounds):
            MyEncrypter.set_user_key(f"CHI2TEST-{round_idx}", image=image_path)
            if num_rounds == 1:
                MyEncrypter.set_user_key()
            encryption_result = MyEncrypter.encrypt(image_path, print_profile=False)
            encrypted_image = encryption_result.encrypted_image
            MyEncrypter.resume_user_key()

            # 计算当轮各通道的直方图及卡方值
            current_chi2 = {}
            if num_channels == 3:
                for i, name in enumerate(["R", "G", "B"]):
                    channel_data = encrypted_image[..., i].ravel()
                    hist = np.bincount(channel_data, minlength=256)
                    current_chi2[name] = calculate_chi_square(hist, channel_data.size)
            else:
                gray_pixels = np.mean(encrypted_image, axis=2).astype(np.uint8).ravel()
                hist = np.bincount(gray_pixels, minlength=256)
                current_chi2["Gray"] = calculate_chi_square(hist, gray_pixels.size)

            chi2_per_round.append(current_chi2)

        # 对多轮的卡方值取平均
        mean_chi2 = {}
        for ch in channel_names:
            values = [one_round[ch] for one_round in chi2_per_round]
            mean_chi2[ch] = float(np.mean(values))

        avg_chi2 = float(
            np.mean([one_round[ch] for one_round in chi2_per_round for ch in channel_names])
        )
        mean_chi2["average"] = avg_chi2

        # 判定是否通过卡方检验：所有通道均通过且平均值通过临界值判定
        is_passed = all(mean_chi2[ch] < critical_value for ch in channel_names)

        run_data.append({
            "name": Path(image_path).name,
            "encrypted_chi2": mean_chi2,
            "channel_names": channel_names,
            "passed": is_passed,
        })

    # ---------- 打印卡方检验表格 ----------
    print("\nChi-Square test analysis")
    print("------------------------")
    print(f"Significance level (α): {alpha}, Critical value (df=255): {critical_value:.2f}")

    channel_order = [c for c in ["R", "G", "B", "Gray"]
                     if any(c in d["encrypted_chi2"] for d in run_data)]

    table_rows = []
    for data in run_data:
        row = [data["name"]]
        for ch in channel_order:
            val = data["encrypted_chi2"].get(ch)
            row.append(f"{val:.2f}" if val is not None else "-")
        row.append(f"{data['encrypted_chi2']['average']:.2f}")
        row.append("Pass" if data["passed"] else "Fail")
        table_rows.append(row)

    # 底部所有图像的统计均值行
    mean_row = ["Mean"]
    all_channel_values = []
    for ch in channel_order:
        values = [d["encrypted_chi2"][ch] for d in run_data if ch in d["encrypted_chi2"]]
        mean_row.append(f"{np.mean(values):.2f}" if values else "-")
        all_channel_values.extend(values)

    total_mean_val = float(np.mean(all_channel_values))
    mean_row.append(f"{total_mean_val:.2f}")
    mean_row.append("Pass" if total_mean_val < critical_value else "Fail")
    table_rows.append(mean_row)

    headers = ("Image",) + tuple(f"{ch} χ²" for ch in channel_order) \
              + ("Average χ²", f"Passed (α={alpha})")
    print_table(headers, table_rows)

    return run_data
# ============================================================
# 一键运行入口
# ============================================================

if __name__ == "__main__":
    run_encryption_decryption_test(show=True)
    
    #size=8 只使用默认的V2秘钥 结果最好
    run_histogram_test(show=False, save_dir=FIGURE_DIR,save_subfigures=False)
    # run_chi_square_test(alpha=0.05)
