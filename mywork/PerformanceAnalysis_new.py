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

from .Encryption import Encrypter, DeEncrypter, EncryptionConfig

# ============================================================
# CONFIG：所有可调参数都在这一坨
# ============================================================

# 加密配置（保持和之前一致）
ENCRYPTION_CONFIG = EncryptionConfig(
    seed=2026,
    b_max=64,  #default=64,  # 最大块大小，越大速度越快，但对小图不友好
    b_min=4,   #default=4,   # 最小块大小，越小速度越慢，但对小图更友好
    block_operation="xor",
    global_parallel_size=8,
    global_permutation_method="global_bit_permutation_v2",
    cross_plane_diffusion = True,
    cross_plane_shift = 1,
    bidirectional_diffusion = True,
)

# 测试图片列表
IMAGE_PATHS = [
    r"C:\ImageEncryption\images\img1.png",
    r"C:\ImageEncryption\images\img2.png",
    r"C:\ImageEncryption\images\img3.png",
    r"C:\ImageEncryption\images\img4.png",
    r"C:\ImageEncryption\images\img5.png",
    # r"C:\ImageEncryption\images\img6.png",  
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

# V3 外部密钥长度（bit）。密钥敏感性测试默认翻转全部 USER_KEY_BITS 位。
USER_KEY_BITS = 512
# 可视化密钥敏感性：把 K 的这一位（0 基）翻转得到 Kt。改这里就能换示范位。
KEY_SENSITIVITY_FLIP_BIT = 0

# 差分攻击（NPCR / UACI）
# N：做 N 轮，第 i 轮翻转像素 (i, i) 的 R 通道最低位（i = 0, 1, ..., N-1）。
DIFF_NUM_TRIALS = 1
# True：P 和 P' 都用 SHA-512(P)+userkey 派生密钥流，用来看「关掉明文哈希」后置乱/扩散还够不够。
DIFF_FREEZE_IMAGE_HASH = False
# 表格按这些边长出列。现有测试图都是 512×512，其他尺寸用最近邻缩放。
DIFF_IMAGE_SIZES = (256, 512, 1024)

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
        num_rounds = 1
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


from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ============================================================
# 测试 4：像素相关性分析（3D 散点图视角校准版 + 相关系数表格）
# ============================================================

def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """计算两组像素序列的皮尔逊相关系数。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0.0
    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    return float(cov / (std_x * std_y))


def get_adjacent_pixel_pairs(channel_data: np.ndarray, direction: str):
    """提取二维单通道图像在指定方向（水平/垂直/对角）上的所有相邻像素对 (x, y)。"""
    if direction == "Horizontal":
        x = channel_data[:, :-1].ravel()
        y = channel_data[:, 1:].ravel()
    elif direction == "Vertical":
        x = channel_data[:-1, :].ravel()
        y = channel_data[1:, :].ravel()
    elif direction == "Diagonal":
        x = channel_data[:-1, :-1].ravel()
        y = channel_data[1:, 1:].ravel()
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return x, y


def sample_pairs(x: np.ndarray, y: np.ndarray, num_samples=2500, seed=42):
    """从相邻像素对中随机采样固定数量，保证 3D 散点图的清晰度与渲染速度。"""
    total = len(x)
    if total <= num_samples:
        return x, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=num_samples, replace=False)
    return x[indices], y[indices]


def run_correlation_test(
    show=True,
    save_dir=None,
    save_subfigures=False,
    plot_mode="by_direction",  # "by_direction" (新版图2样式) 或 "by_channel" (原版样式)
    num_samples=2500,
    dpi=150,
):
    """测试算法的相邻像素相关性系数，并生成三维切片散点图与数据表格。

    参数:
        show: 是否弹窗展示生成的 3D 数据图。
        save_dir: 拼接总图的保存目录，为 None 则不保存整图。
        save_subfigures: 是否在 'figures/' 下新建独立目录并将各子图逐一单独导出保存。
        plot_mode: 绘图方式。
            - "by_direction": 子图按 H/V/D 排列，每个子图内部展示 R/G/B 三个切面（图2样式）；
            - "by_channel": 子图按 R/G/B 排列，每个子图内部展示 H/V/D 三个切面（原版样式）。
        num_samples: 3D 散点图每个切面的随机抽样点数，默认 2500。
        dpi: 图像保存的分辨率。
    """
    directions = ["Horizontal", "Vertical", "Diagonal"]
    channels = ["R", "G", "B"]

    # 保持原配色：R (红), G (绿), B (蓝)
    channel_colors = {
        "R": "#b82424",
        "G": "#2b8236",
        "B": "#1e6bb8",
    }
    dir_colors = {
        "Horizontal": "#b82424",
        "Vertical": "#2b8236",
        "Diagonal": "#1e6bb8",
    }

    # 各方向对应的相邻像素 Z 轴标题
    z_labels = {
        "Horizontal": "Pixel value at location(x+1,y)",
        "Vertical": "Pixel value at location(x,y+1)",
        "Diagonal": "Pixel value at location(x+1,y+1)",
    }

    table_data = []
    figures = []

    for img_idx, image_path in enumerate(IMAGE_PATHS):
        img_name = Path(image_path).name
        stem = Path(image_path).stem

        # ---------- 1. 加密（默认密钥，单轮） ----------
        MyEncrypter.set_user_key(image=image_path)
        encryption_result = MyEncrypter.encrypt(image_path, print_profile=False)
        encrypted_image = encryption_result.encrypted_image
        MyEncrypter.resume_user_key()

        original_image = load_rgb_image(image_path)

        # ---------- 2. 计算各方向、各通道密文相关系数 ----------
        row_metrics = {"name": img_name}
        for d in directions:
            for c_idx, c_name in enumerate(channels):
                x_cipher, y_cipher = get_adjacent_pixel_pairs(encrypted_image[..., c_idx], d)
                r_val = calculate_correlation(x_cipher, y_cipher)
                row_metrics[f"{d}_{c_name}"] = r_val
        table_data.append(row_metrics)

        # ---------- 3. 绘制三维切片散点图 ----------
        fig = plt.figure(figsize=(15.5, 9.6))
        target_images = [
            (original_image, "plaintext"),
            (encrypted_image, "ciphertext"),
        ]

        sub_axes = []
        plot_idx = 1

        # ==================== 绘图分支 1：图 2 样式 (by_direction) ====================
        if plot_mode == "by_direction":
            channel_positions = {"R": 0, "G": 1, "B": 2}

            for row_type_idx, (img_data, img_type) in enumerate(target_images):
                for d_idx, d_name in enumerate(directions):
                    ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
                    sub_axes.append((ax, f"{d_name}_{img_type}"))

                    # 在当前方向的 3D 坐标系内，依次绘制 R、G、B 三个平行切面
                    for c_idx, c_name in enumerate(channels):
                        x_pairs, y_pairs = get_adjacent_pixel_pairs(img_data[..., c_idx], d_name)
                        sx, sy = sample_pairs(x_pairs, y_pairs, num_samples=num_samples, seed=42 + img_idx)

                        # 切片轴映射为 X 轴 (0: R, 1: G, 2: B)
                        # 像素值映射为 Y 轴向左前方倾斜延伸，Z 轴垂直向上延伸
                        coord_slice = np.full_like(sx, channel_positions[c_name])
                        coord_px = sx
                        coord_adj = sy

                        ax.scatter(
                            coord_slice, coord_px, coord_adj,
                            s=0.6,
                            c=channel_colors[c_name],
                            alpha=0.35,
                            edgecolors="none",
                            rasterized=True,
                        )

                    # 坐标范围与等比立方比例
                    ax.set_box_aspect((1.1, 1.0, 1.0))
                    ax.set_xlim(-0.25, 2.35)
                    ax.set_ylim(0, 255)
                    ax.set_zlim(0, 255)

                    # 切片轴刻度显示为 R, G, B
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(["R", "G", "B"], fontsize=8.5)

                    # 像素轴 (X 轴) 刻度与标题
                    ax.set_yticks([0, 100, 200])
                    ax.set_yticklabels([0, 100, 200], fontsize=8)
                    ax.set_ylabel("Pixel value at location(x,y)", fontsize=8.5, labelpad=4)

                    # 垂直高度轴 (Z 轴) 刻度与标题
                    ax.set_zticks([0, 50, 100, 150, 200, 250])
                    ax.set_zticklabels([0, 50, 100, 150, 200, 250], fontsize=8)
                    ax.set_zlabel(z_labels[d_name], fontsize=8.5, labelpad=2)

                    ax.tick_params(axis="x", pad=-2)
                    ax.tick_params(axis="y", pad=-2)
                    ax.tick_params(axis="z", pad=-1)

                    # 纯白背景：隐藏背景面和网格线
                    ax.grid(False)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor("none")
                    ax.yaxis.pane.set_edgecolor("none")
                    ax.zaxis.pane.set_edgecolor("none")

                    ax.view_init(elev=26, azim=-128)

                    plot_idx += 1

            plt.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06, wspace=0.18, hspace=0.22)

        # ==================== 绘图分支 2：原版样式 (by_channel) ====================
        elif plot_mode == "by_channel":
            dir_positions = {"Horizontal": 0, "Vertical": 1, "Diagonal": 2}

            for row_type_idx, (img_data, img_type) in enumerate(target_images):
                for c_idx, c_name in enumerate(channels):
                    ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
                    sub_axes.append((ax, f"{c_name}_{img_type}"))

                    for d in directions:
                        x_pairs, y_pairs = get_adjacent_pixel_pairs(img_data[..., c_idx], d)
                        sx, sy = sample_pairs(x_pairs, y_pairs, num_samples=num_samples, seed=42 + img_idx)

                        coord_x = np.full_like(sx, dir_positions[d])
                        coord_y = sx
                        coord_z = sy

                        ax.scatter(
                            coord_x, coord_y, coord_z,
                            s=0.6,
                            c=dir_colors[d],
                            alpha=0.35,
                            edgecolors="none",
                            rasterized=True,
                        )

                    ax.set_box_aspect((1.1, 1.0, 1.0))
                    ax.set_xlim(-0.25, 2.35)
                    ax.set_ylim(0, 255)
                    ax.set_zlim(0, 255)

                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(directions, fontsize=8.5)
                    ax.set_yticks([0, 100, 200])
                    ax.set_yticklabels([0, 100, 200], fontsize=8)
                    ax.set_zticks([0, 50, 100, 150, 200, 250])
                    ax.set_zticklabels([0, 50, 100, 150, 200, 250], fontsize=8)

                    ax.tick_params(axis="x", pad=-2)
                    ax.tick_params(axis="y", pad=-2)
                    ax.tick_params(axis="z", pad=-1)

                    ax.grid(False)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor("none")
                    ax.yaxis.pane.set_edgecolor("none")
                    ax.zaxis.pane.set_edgecolor("none")

                    ax.view_init(elev=26, azim=-128)

                    plot_idx += 1

            plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.05, wspace=0.06, hspace=0.16)
        else:
            raise ValueError(f"Unknown plot_mode: {plot_mode}. Must be 'by_direction' or 'by_channel'.")

        figures.append((stem, fig))

        # ---------- 保存各个独立子图 ----------
        if save_subfigures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subfig_dir = Path("figures") / f"correlation_{plot_mode}_{timestamp}" / stem
            subfig_dir.mkdir(parents=True, exist_ok=True)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax, name_tag in sub_axes:
                bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                out_path = subfig_dir / f"{stem}_{name_tag}.png"
                fig.savefig(out_path, bbox_inches=bbox, dpi=dpi, facecolor="white")
            print(f"[{stem}] saved individual subfigures to: {subfig_dir}")

        # ---------- 保存整张大图 ----------
        if save_dir is not None:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path = out_dir / f"{stem}_correlation_{plot_mode}.png"
            fig.savefig(fig_path, dpi=dpi, facecolor="white")
            print(f"saved figure: {fig_path}")

    # ============================================================
    # 打印相关系数表格（6位小数精度）
    # ============================================================
    print("\nCorrelation coefficients analysis")
    print("---------------------------------")

    table_rows = []
    for item in table_data:
        row = [item["name"]]
        for d in directions:
            for c in channels:
                val = item[f"{d}_{c}"]
                row.append(f"{val:.6f}")
        table_rows.append(row)

    mean_row = ["Channel Mean"]
    channel_means = {}
    for d in directions:
        for c in channels:
            vals = [item[f"{d}_{c}"] for item in table_data]
            avg_c = float(np.mean(vals))
            channel_means[f"{d}_{c}"] = avg_c
            mean_row.append(f"{avg_c:.6f}")
    table_rows.append(mean_row)

    dir_mean_h = float(np.mean([channel_means[f"Horizontal_{c}"] for c in channels]))
    dir_mean_v = float(np.mean([channel_means[f"Vertical_{c}"] for c in channels]))
    dir_mean_d = float(np.mean([channel_means[f"Diagonal_{c}"] for c in channels]))

    direction_mean_row = [
        "Direction Mean",
        "-", f"{dir_mean_h:.6f}", "-",
        "-", f"{dir_mean_v:.6f}", "-",
        "-", f"{dir_mean_d:.6f}", "-",
    ]
    table_rows.append(direction_mean_row)

    headers = (
        "Images",
        "H (R)", "H (G)", "H (B)",
        "V (R)", "V (G)", "V (B)",
        "D (R)", "D (G)", "D (B)",
    )
    print_table(headers, table_rows)
    print(f">> Direction Overall Means | Horizontal: {dir_mean_h:.6f} | Vertical: {dir_mean_v:.6f} | Diagonal: {dir_mean_d:.6f}\n")

    if show:
        plt.show()

    return table_data, figures


# ============================================================
# 测试 5：明文图像与密文图像的信息熵
# ============================================================

def calculate_information_entropy(image):
    """计算 8-bit 图像的信息熵，RGB 图像按全部通道像素联合统计。"""
    image_array = np.asarray(image, dtype=np.uint8)
    counts = np.bincount(image_array.ravel(), minlength=256)
    probabilities = counts[counts > 0] / image_array.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def run_information_entropy_test(num_rounds=None):
    """每张图循环 num_rounds 轮不同 user_key 加密，信息熵取均值后制表。

    明文信息熵与密钥无关，只算一次；密文信息熵每轮各算一次，最后取多轮均值。
    这样单轮偶然偏高或偏低的结果不会直接进入论文表格。
    """
    if num_rounds is None:
        num_rounds = NUM_USER_KEY_ROUNDS
    if num_rounds <= 0:
        raise ValueError("NUM_USER_KEY_ROUNDS must be a positive integer")

    run_data = []

    for image_path in IMAGE_PATHS:
        original_image = np.asarray(pil_image.open(image_path).convert("RGB"))
        plaintext_entropy = calculate_information_entropy(original_image)
        num_rounds = 1
        ciphertext_entropy_per_round = []
        for round_idx in range(num_rounds):
            MyEncrypter.set_user_key(f"ENTROPYTEST-{round_idx}", image=image_path)
            encryption_result = MyEncrypter.encrypt(image_path, print_profile=False)
            encrypted_image = encryption_result.encrypted_image
            MyEncrypter.resume_user_key()

            ciphertext_entropy_per_round.append(
                calculate_information_entropy(encrypted_image)
            )

        run_data.append({
            "name": Path(image_path).name,
            "plaintext_entropy": plaintext_entropy,
            "ciphertext_entropy": float(np.mean(ciphertext_entropy_per_round)),
            "ciphertext_entropy_per_round": ciphertext_entropy_per_round,
        })

    # ---------- 打印信息熵表格 ----------
    print("\nInformation entropy analysis")
    print("----------------------------")
    print(f"User key rounds: {num_rounds}")

    table_rows = [
        (
            data["name"],
            f'{data["plaintext_entropy"]:.6f}',
            f'{data["ciphertext_entropy"]:.6f}',
        )
        for data in run_data
    ]

    mean_row = [
        "Mean",
        f'{np.mean([data["plaintext_entropy"] for data in run_data]):.6f}',
        f'{np.mean([data["ciphertext_entropy"] for data in run_data]):.6f}',
    ]
    table_rows.append(mean_row)

    print_table(
        ("Images", "Plaintext Entropy", "Ciphertext Entropy"),
        table_rows,
    )

    return run_data


# ============================================================
# 测试 6：密钥敏感性（翻转 512 位 V3 用户密钥的每一位）
# ============================================================

def calculate_nbcr(reference, candidate):
    """NBCR(%)：两张图所有 uint8 位的汉明距离占比，约 50% 表示互不相关。"""
    a = np.asarray(reference, dtype=np.uint8)
    b = np.asarray(candidate, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"NBCR requires identical shapes: {a.shape} != {b.shape}")
    changed_bits = int(np.unpackbits(np.bitwise_xor(a, b).ravel()).sum())
    return changed_bits / (a.size * 8) * 100.0


def run_key_sensitivity_visual_test(
    show=True,
    save_dir=None,
    image_path=None,
    flip_bit=None,
    dpi=150,
):
    """只翻转 1 位密钥，画 2x3 密钥敏感性示意（不扫 512 位，很快）。

    第一行（加密）：原文、K 的密文、Kt 的密文。
    第二行（解密）：K 的密文、用 K 解密、用 Kt 解密。
    Kt 由 K 翻转 KEY_SENSITIVITY_FLIP_BIT 这一位得到。

    错误密钥解密必须走 Decrypt_V2：普通 decrypt 会读 metadata 里的正确密钥材料，
    看起来就像“错密钥也能解开”。
    """
    from .KeyStream import normalize_user_key_v3

    if image_path is None:
        image_path = IMAGE_PATHS[0]
    if flip_bit is None:
        flip_bit = KEY_SENSITIVITY_FLIP_BIT
    image_path = Path(image_path)
    flip_bit = int(flip_bit)
    if not 0 <= flip_bit < USER_KEY_BITS:
        raise ValueError(f"flip_bit must be in [0, {USER_KEY_BITS - 1}]")

    original_image = load_rgb_image(image_path)
    key_k = normalize_user_key_v3(None)
    flipped = "1" if key_k[flip_bit] == "0" else "0"
    key_kt = key_k[:flip_bit] + flipped + key_k[flip_bit + 1:]

    # ---------- 用 K 加密，得到 C_K 和 metadata ----------
    MyEncrypter.set_user_key(key_k, image=image_path)
    result_k = MyEncrypter.encrypt(image_path, print_profile=False)
    cipher_k = result_k.encrypted_image
    metadata = result_k.metadata
    MyEncrypter.resume_user_key()

    # 正确密钥解密走 metadata，应无损恢复原文
    plain_k = MyDeEncrypter.decrypt(
        cipher_k, metadata, print_profile=False
    ).decrypted_image

    # ---------- 用 Kt：先错密钥解密 C_K，再错密钥加密同一明文 ----------
    # encrypt 会把 _cml 清掉，所以必须先 Decrypt_V2 再 encrypt。
    MyEncrypter.set_user_key(key_kt, image=image_path)
    plain_kt = MyDeEncrypter.Decrypt_V2(
        cipher_k, MyEncrypter, metadata, print_profile=False
    ).decrypted_image
    cipher_kt = MyEncrypter.encrypt(image_path, print_profile=False).encrypted_image
    MyEncrypter.resume_user_key()

    exact_recovery = bool(np.array_equal(original_image, plain_k))
    cipher_nbcr = calculate_nbcr(cipher_k, cipher_kt)
    wrong_decrypt_nbcr = calculate_nbcr(original_image, plain_kt)

    print("\nKey sensitivity visual check")
    print("----------------------------")
    print(f"Image: {image_path.name}")
    print(f"Flipped bit (0-based / 1-based): {flip_bit} / {flip_bit + 1}")
    print_table(
        ("Item", "Value"),
        [
            ("Exact recovery with K", "Yes" if exact_recovery else "No"),
            ("NBCR(C_K, C_Kt) %", f"{cipher_nbcr:.6f}"),
            ("NBCR(P, P_Kt) %", f"{wrong_decrypt_nbcr:.6f}"),
        ],
    )

    # ---------- 2x3 图 ----------
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 6.2))
    figure.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.06, wspace=0.06, hspace=0.16)
    panels = (
        (0, 0, original_image, r"(a) Original image"),
        (0, 1, cipher_k, r"(b) $C_{K}$"),
        (0, 2, cipher_kt, r"(c) $C_{K_t}$"),
        (1, 0, cipher_k, r"(d) $C_{K}$"),
        (1, 1, plain_k, r"(e) $P_{K}$"),
        (1, 2, plain_kt, r"(f) $P_{K_t}$"),
    )
    for row, col, image, caption in panels:
        axis = axes[row, col]
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        # axis.set_xlabel(caption, fontsize=12, labelpad=6)

    # axes[0, 0].text(
    #     -0.10, 0.5, "Encryption",
    #     transform=axes[0, 0].transAxes,
    #     rotation=90, ha="right", va="center", fontsize=12,
    # )
    # axes[1, 0].text(
    #     -0.10, 0.5, "Decryption",
    #     transform=axes[1, 0].transAxes,
    #     rotation=90, ha="right", va="center", fontsize=12,
    # )

    fig_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "key_sensitivity_visual.png"
        figure.savefig(fig_path, dpi=dpi, facecolor="white")
        print(f"saved figure: {fig_path}")

    if show:
        plt.show()

    return {
        "image": str(image_path),
        "flip_bit": flip_bit,
        "exact_recovery": exact_recovery,
        "cipher_nbcr_percent": cipher_nbcr,
        "wrong_decrypt_nbcr_percent": wrong_decrypt_nbcr,
        "figure": figure,
        "save_path": None if fig_path is None else str(fig_path),
    }


def run_key_sensitivity_test(
    show=True,
    save_dir=None,
    image_path=None,
    bit_indices=None,
    dpi=150,
):
    """逐个翻转 512 位 V3 用户密钥的每一位，画 NBCR 曲线并打印统计表。

    左图 NBCR(C1, C2)：同一明文分别用正确密钥、翻转第 i 位的错误密钥加密。
    右图 NBCR(D1, D2)：同一密文 C1 分别用正确密钥、错误密钥解密；D1 就是明文。
    两条曲线都应紧贴 0.5，越接近 50% 说明一位密钥变化就足以让结果完全改变。

    参数:
        show: 是否 plt.show() 弹窗展示。
        save_dir: 曲线图保存目录，为 None 则不保存。
        image_path: 测试图片，默认取 IMAGE_PATHS 的第一张。
        bit_indices: 要翻转的密钥位（0 基）。默认全部 USER_KEY_BITS 位；传入少量下标可快速自检。
        dpi: 保存图片分辨率。

    注意: 512 位每位各需 1 次加密 + 1 次解密，整轮耗时较长（见返回值的 seconds）。
    表格打印百分比（≈50%），曲线按参考图以小数绘制（0.49-0.51）。
    """
    from .KeyStream import normalize_user_key_v3

    if image_path is None:
        image_path = IMAGE_PATHS[0]
    image_path = Path(image_path)
    original_image = load_rgb_image(image_path)

    if bit_indices is None:
        bit_indices = range(USER_KEY_BITS)
    bit_indices = [int(index) for index in bit_indices]
    for bit_index in bit_indices:
        if not 0 <= bit_index < USER_KEY_BITS:
            raise ValueError(
                f"bit_indices must be in the zero-based range [0, {USER_KEY_BITS - 1}]"
            )

    baseline_key = normalize_user_key_v3(None)

    # ---------- 基准：正确密钥加密同一明文 ----------
    start_time = time.perf_counter()
    MyEncrypter.set_user_key(baseline_key, image=image_path)
    baseline_result = MyEncrypter.encrypt(image_path, print_profile=False)
    baseline_cipher = baseline_result.encrypted_image
    baseline_metadata = baseline_result.metadata
    MyEncrypter.resume_user_key()

    cipher_nbcr = []
    decryption_nbcr = []

    for bit_index in bit_indices:
        flipped_bit = "1" if baseline_key[bit_index] == "0" else "0"
        wrong_key = (
            baseline_key[:bit_index] + flipped_bit + baseline_key[bit_index + 1:]
        )

        # 错误密钥解密 C1：必须先解密再加密，set_user_key 会重建 CML。
        MyEncrypter.set_user_key(wrong_key, image=image_path)
        wrong_decryption = MyDeEncrypter.Decrypt_V2(
            baseline_cipher, MyEncrypter, baseline_metadata, print_profile=False
        ).decrypted_image
        MyEncrypter.resume_user_key()

        # 错误密钥加密同一明文
        MyEncrypter.set_user_key(wrong_key, image=image_path)
        wrong_cipher = MyEncrypter.encrypt(image_path, print_profile=False).encrypted_image
        MyEncrypter.resume_user_key()

        cipher_nbcr.append(calculate_nbcr(baseline_cipher, wrong_cipher))
        decryption_nbcr.append(calculate_nbcr(original_image, wrong_decryption))

    elapsed_seconds = time.perf_counter() - start_time

    # ---------- 打印统计表 ----------
    print("\nKey sensitivity analysis")
    print("------------------------")
    print(f"Image: {Path(image_path).name}, flipped key bits: {len(bit_indices)}/{USER_KEY_BITS}")
    print(f"Elapsed: {elapsed_seconds:.2f} s")

    def summary_row(label, values):
        values = np.asarray(values, dtype=np.float64)
        return (
            label,
            f"{np.mean(values):.6f}",
            f"{np.min(values):.6f}",
            f"{np.max(values):.6f}",
            f"{np.mean(np.abs(values - 50.0)):.6f}",
        )

    table_rows = [
        summary_row("NBCR (C1, C2)", cipher_nbcr),
        summary_row("NBCR (D1, D2)", decryption_nbcr),
    ]
    print_table(
        ("Metric", "Mean (%)", "Min (%)", "Max (%)", "Mean |Diff-50|"),
        table_rows,
    )

    # 完全不变的位不是"敏感性略差"，而是密钥派生根本没用到这一位，必须显式报出来。
    for label, values in (("NBCR (C1, C2)", cipher_nbcr), ("NBCR (D1, D2)", decryption_nbcr)):
        insensitive = [
            bit_index + 1
            for bit_index, value in zip(bit_indices, values)
            if value == 0.0
        ]
        if insensitive:
            print(
                f"WARNING: {len(insensitive)} key bit(s) left the result bit-identical "
                f"({label}) -> dead key bits: {insensitive}"
            )

    # ---------- 画 1x2 NBCR 曲线（与参考图一致，纵轴为小数） ----------
    x_values = [index + 1 for index in bit_indices]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), squeeze=False)

    panels = (
        (axes[0, 0], cipher_nbcr, "#00a2e8", r"$NBCR(C_1(i),C_2(i))$"),
        (axes[0, 1], decryption_nbcr, "#ff0000", r"$NBCR(D_1(i),D_2(i))$"),
    )
    plotted = []
    for axis, values, color, label in panels:
        fractions = np.asarray(values, dtype=np.float64) / 100.0
        plotted.extend(fractions.tolist())
        axis.plot(x_values, fractions, color=color, linewidth=0.9)
        axis.set_ylabel(label, fontsize=11)
        axis.set_xlabel(f"Secret key(1-{USER_KEY_BITS} bit)", fontsize=10)
        axis.set_ylim(0.49, 0.51)
        axis.set_yticks(np.arange(0.49, 0.5101, 0.002))
        axis.set_xticks(list(range(0, USER_KEY_BITS + 1, 64)))
        axis.set_xlim(0, USER_KEY_BITS)
        axis.tick_params(labelsize=9)
        axis.grid(False)

    figure.tight_layout()

    if plotted and (min(plotted) < 0.49 or max(plotted) > 0.51):
        print("WARNING: some NBCR values fall outside the 0.49-0.51 window "
              "and are clipped in the figure; check the table above.")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "key_sensitivity.png"
        figure.savefig(fig_path, dpi=dpi, facecolor="white")
        print(f"saved figure: {fig_path}")

    if show:
        plt.show()

    return {
        "image": str(image_path),
        "bit_indices": bit_indices,
        "cipher_nbcr_percent": cipher_nbcr,
        "decryption_nbcr_percent": decryption_nbcr,
        "mean_cipher_nbcr_percent": float(np.mean(cipher_nbcr)),
        "mean_decryption_nbcr_percent": float(np.mean(decryption_nbcr)),
        "elapsed_seconds": elapsed_seconds,
        "figure": figure,
    }


# ============================================================
# 测试 8：差分攻击（NPCR / UACI）
# ============================================================

IDEAL_NPCR_PERCENT = 99.6094
IDEAL_UACI_PERCENT = 33.4635


def calculate_npcr(cipher_a, cipher_b):
    """NPCR(%)：两幅密文对应位置像素不同的比例。理想值约 99.6094%。"""
    a = np.asarray(cipher_a, dtype=np.uint8)
    b = np.asarray(cipher_b, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError(f"NPCR requires identical shapes: {a.shape} != {b.shape}")
    return float(np.mean(a != b) * 100.0)


def calculate_uaci(cipher_a, cipher_b):
    """UACI(%)：两幅密文对应像素绝对差 / 255 的均值。理想值约 33.4635%。"""
    a = np.asarray(cipher_a, dtype=np.int16)
    b = np.asarray(cipher_b, dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"UACI requires identical shapes: {a.shape} != {b.shape}")
    return float(np.mean(np.abs(a - b)) / 255.0 * 100.0)


def resize_square_rgb(image, size):
    """把 RGB 图缩放到 size×size。边长已相同则只 copy，避免无意义的重采样。"""
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image.copy())
    resized = pil_image.fromarray(image).resize(
        (size, size), resample=pil_image.Resampling.NEAREST
    )
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def run_differential_attack_test(
    num_trials=None,
    freeze_image_hash=None,
    image_sizes=None,
):
    """明文差 1 bit 后比较两幅密文的 NPCR / UACI。

    每张图、每个尺寸：
      1. 先把图缩放到该尺寸，再加密得到 C1（只加密一次）。
      2. 做 N 轮：第 i 轮把像素 (i, i) 的 R 通道最低位翻转，加密得到 C2_i。
      3. 对该尺寸的 N 轮 NPCR / UACI 取平均，填进表格。

    freeze_image_hash=False（默认，真实系统）：
      P' 用 SHA-512(P')+userkey，明文哈希会跟着变。
    freeze_image_hash=True（关掉明文哈希）：
      P 和 P' 都用 SHA-512(P)+userkey，只看置乱/扩散本身的雪崩。

    参数
    ----
    num_trials : N，轮数。None 则用顶部 DIFF_NUM_TRIALS。
    freeze_image_hash : None 则用顶部 DIFF_FREEZE_IMAGE_HASH。
    image_sizes : 例如 (256, 512, 1024)。None 则用顶部 DIFF_IMAGE_SIZES。
    """
    if num_trials is None:
        num_trials = DIFF_NUM_TRIALS
    if freeze_image_hash is None:
        freeze_image_hash = DIFF_FREEZE_IMAGE_HASH
    if image_sizes is None:
        image_sizes = DIFF_IMAGE_SIZES

    num_trials = int(num_trials)
    if num_trials <= 0:
        raise ValueError("num_trials (N) must be a positive integer")
    image_sizes = tuple(int(size) for size in image_sizes)
    if not image_sizes:
        raise ValueError("image_sizes must not be empty")
    for size in image_sizes:
        if size <= 0:
            raise ValueError(f"image size must be positive, got {size}")
        if num_trials > size:
            raise ValueError(
                f"N={num_trials} needs pixels (0,0)..({num_trials - 1},{num_trials - 1}), "
                f"but size {size} is too small"
            )

    started = time.perf_counter()
    run_data = []

    for image_path in IMAGE_PATHS:
        original = load_rgb_image(image_path)
        size_results = {}

        for size in image_sizes:
            plain = resize_square_rgb(original, size)

            # C1：未改动的明文，每个尺寸只加密一次，后面 N 轮都跟它比。
            MyEncrypter.set_user_key(image=plain)
            cipher_p = MyEncrypter.encrypt(plain, print_profile=False).encrypted_image
            MyEncrypter.resume_user_key()

            npcr_rounds = []
            uaci_rounds = []
            for trial_index in range(num_trials):
                modified = plain.copy()
                modified[trial_index, trial_index, 0] ^= np.uint8(1)

                # False：P' 自己的 SHA-512；True：强制仍用 P 的 SHA-512。
                hash_source = plain if freeze_image_hash else modified
                MyEncrypter.set_user_key(image=hash_source)
                cipher_t = MyEncrypter.encrypt(modified, print_profile=False).encrypted_image
                MyEncrypter.resume_user_key()

                npcr_rounds.append(calculate_npcr(cipher_p, cipher_t))
                uaci_rounds.append(calculate_uaci(cipher_p, cipher_t))

            size_results[size] = {
                "npcr_percent": float(np.mean(npcr_rounds)),
                "uaci_percent": float(np.mean(uaci_rounds)),
                "npcr_per_trial": npcr_rounds,
                "uaci_per_trial": uaci_rounds,
            }

        run_data.append({
            "name": Path(image_path).name,
            "sizes": size_results,
        })

    elapsed_seconds = time.perf_counter() - started

    print("\nDifferential attack analysis (NPCR / UACI)")
    print("------------------------------------------")
    print(f"Trials N: {num_trials}  (flip R-channel LSB at pixel (i, i), i = 0..{num_trials - 1})")
    print(
        "Freeze image hash: "
        f"{freeze_image_hash}  "
        "(True = both P and P' use SHA-512(P)+userkey)"
    )
    print(f"Ideal NPCR = {IDEAL_NPCR_PERCENT:.4f}%, ideal UACI = {IDEAL_UACI_PERCENT:.4f}%")
    print(f"Elapsed: {elapsed_seconds:.2f} s")

    headers = ["Image"]
    for size in image_sizes:
        headers.append(f"{size}×{size} NPCR")
        headers.append(f"{size}×{size} UACI")

    table_rows = []
    for data in run_data:
        row = [data["name"]]
        for size in image_sizes:
            row.append(f'{data["sizes"][size]["npcr_percent"]:.4f}')
            row.append(f'{data["sizes"][size]["uaci_percent"]:.4f}')
        table_rows.append(row)

    mean_row = ["Mean"]
    for size in image_sizes:
        mean_row.append(
            f'{np.mean([data["sizes"][size]["npcr_percent"] for data in run_data]):.4f}'
        )
        mean_row.append(
            f'{np.mean([data["sizes"][size]["uaci_percent"] for data in run_data]):.4f}'
        )
    table_rows.append(mean_row)
    print_table(tuple(headers), table_rows)

    return {
        "num_trials": num_trials,
        "freeze_image_hash": bool(freeze_image_hash),
        "image_sizes": image_sizes,
        "elapsed_seconds": elapsed_seconds,
        "rows": run_data,
    }


if __name__ == "__main__":
    # run_encryption_decryption_test(show=True)
    
    #size=8 只使用默认的V3秘钥 结果最好
    # run_histogram_test(show=False, save_dir=FIGURE_DIR,save_subfigures=False)
    # run_chi_square_test(alpha=0.05)
    
    # run_information_entropy_test()
    
    # run_correlation_test(
    #     show=True,
    #     save_dir=FIGURE_DIR,
    #     save_subfigures=True,
    #     num_samples=10000,
    #     plot_mode="by_direction",
    # )
    # run_key_sensitivity_test(
    #     show=True,
    #     save_dir=FIGURE_DIR,
    #     image_path=IMAGE_PATHS[0],
    #     bit_indices=None,  # None 表示全部 512 位 V3 密钥
    #     dpi=150,
    # )
    # run_key_sensitivity_visual_test(
    #     show=True,
    #     save_dir=FIGURE_DIR,
    #     image_path=IMAGE_PATHS[2],
    #     # flip_bit=0,  # 不写则用顶部 KEY_SENSITIVITY_FLIP_BIT
    #     dpi=150,
    # )
    run_differential_attack_test(
        num_trials=10,            # 不写则用顶部 DIFF_NUM_TRIALS（N）
        freeze_image_hash=False, # 不写则用顶部 DIFF_FREEZE_IMAGE_HASH
        image_sizes=(256, 512, 1024),
    )
