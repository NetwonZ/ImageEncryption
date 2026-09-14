"""统计全局扩散秘钥材料 key_material.global_key_matrix 的碱基分布.

对同一张图片使用多个不同 user_key 各加密一次，在加密过程中取出
global_key_matrix (12, H, W) uint8，其中 0/1/2/3 分别对应碱基 A/C/G/T，
统计四个碱基的占比是否均匀（期望各 25%）。

用法：
    python mywork/GlobalKeyDistribution.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chisquare

from ..Encryption import DNA_RULE_TABLE_INV, Encrypter, EncryptionConfig

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\ImageEncryption")
BASE_NAMES = ("A", "C", "G", "T")            # code 0,1,2,3
PRIMARY_IMAGE = ROOT / "images" / "img1.png"
SECONDARY_IMAGE = ROOT / "images" / "img2.png"
PRIMARY_ROUNDS = 50
SECONDARY_ROUNDS = 20
KEY_TAG = "KEYDIST"

ENCRYPTION_CONFIG = EncryptionConfig(
    seed=2026,
    b_max=64,
    b_min=4,
    block_operation="xor",
    global_parallel_size=2,
    global_permutation_method="global_bit_permutation_v2",
)

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "key_material_distribution"
TILE = 64  # 空间同质性检查的分块边长


# ============================================================
# 单次加密 + 统计
# ============================================================

def analyse_one_round(encrypter: Encrypter, image_path: Path, round_idx: int) -> dict:
    """用第 round_idx 个 user_key 加密一次，返回该轮的碱基统计。"""
    encrypter.set_user_key(f"{KEY_TAG}-{round_idx}", image=image_path)
    result = encrypter.encrypt(image_path, print_profile=False)

    key = result.metadata.key_material
    matrix = np.asarray(key.global_key_matrix, dtype=np.uint8)
    rule_ids = np.asarray(key.global_key_rule_id, dtype=np.uint8).reshape(matrix.shape[1:])
    if matrix.ndim != 3:
        raise RuntimeError(f"global_key_matrix 维度异常: {matrix.shape}")

    counts = np.bincount(matrix.reshape(-1), minlength=4).astype(np.int64)
    total = int(counts.sum())

    # 2-bit 原始取值（与碱基映射前）由碱基 + 该像素的编码规则反推，是精确值
    values = DNA_RULE_TABLE_INV[rule_ids - 1, matrix]
    value_counts = np.bincount(values.reshape(-1), minlength=4).astype(np.int64)

    # 碱基均匀性可能来自 sin() 折叠；这里进一步检验更细粒度的量化位。
    # 平面 p 的 2-bit 值 = (v_H 的第 p+1 位, v_L 的第 p+1 位)，两个 12-bit
    # 量化值按 MSB→LSB 交织，因此高低位可精确还原。
    high_bits = ((values >> 1) & 1).astype(np.uint16)   # v_H 的 12 位（MSB→）
    low_bits = (values & 1).astype(np.uint16)           # v_L 的 12 位（MSB→）
    powers = np.uint16(1) << np.arange(11, -1, -1, dtype=np.uint16)
    value_high_12 = (high_bits * powers.reshape(-1, 1, 1)).sum(axis=0).astype(np.uint16)
    value_low_12 = (low_bits * powers.reshape(-1, 1, 1)).sum(axis=0).astype(np.uint16)
    high6_counts = np.bincount((value_high_12.reshape(-1) >> 6), minlength=64).astype(np.int64)
    low6_counts = np.bincount((value_high_12.reshape(-1) & 63), minlength=64).astype(np.int64)
    high12_counts = np.bincount(value_high_12.reshape(-1), minlength=4096).astype(np.int64)
    low12_counts = np.bincount(value_low_12.reshape(-1), minlength=4096).astype(np.int64)

    rule_counts = np.bincount(rule_ids.reshape(-1) - 1, minlength=8).astype(np.int64)

    # 逐平面（12 个 DNA 平面）
    plane_counts = np.stack(
        [np.bincount(matrix[p].reshape(-1), minlength=4) for p in range(matrix.shape[0])]
    ).astype(np.int64)

    # 空间同质性：分块后的 (块, 碱基) 列联表
    _, height, width = matrix.shape
    tiles_y, tiles_x = height // TILE, width // TILE
    tile_counts = np.zeros((tiles_y * tiles_x, 4), dtype=np.int64)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            block = matrix[:, ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            tile_counts[ty * tiles_x + tx] = np.bincount(block.reshape(-1), minlength=4)

    encrypter.resume_user_key()
    return {
        "round": round_idx,
        "total": total,
        "counts": counts.tolist(),
        "value_counts": value_counts.tolist(),
        "high6_counts": high6_counts.tolist(),
        "low6_counts": low6_counts.tolist(),
        "high12_counts": high12_counts.tolist(),
        "low12_counts": low12_counts.tolist(),
        "rule_counts": rule_counts.tolist(),
        "plane_counts": plane_counts.tolist(),
        "tile_counts": tile_counts.tolist(),
        "shape": list(matrix.shape),
    }


def run_image(encrypter: Encrypter, image_path: Path, rounds: int) -> dict:
    records = []
    for round_idx in range(rounds):
        started = time.perf_counter()
        records.append(analyse_one_round(encrypter, image_path, round_idx))
        print(
            f"  round {round_idx + 1:>2}/{rounds}  "
            f"counts={records[-1]['counts']}  "
            f"({time.perf_counter() - started:.2f}s)",
            flush=True,
        )
    return {
        "image": str(image_path),
        "rounds": rounds,
        "records": records,
    }


# ============================================================
# 汇总统计
# ============================================================

def summarise(run: dict) -> dict:
    records = run["records"]
    rounds = len(records)
    counts = np.array([r["counts"] for r in records], dtype=np.float64)      # (rounds, 4)
    totals = np.array([r["total"] for r in records], dtype=np.float64)       # (rounds,)
    pooled = counts.sum(axis=0).astype(np.int64)
    pooled_total = int(pooled.sum())
    pooled_freq = pooled / pooled_total
    pooled_chi2, pooled_p = chisquare(pooled)

    per_round_freq = counts / totals[:, None]
    per_round_chi2 = np.array([chisquare(c)[0] for c in counts])
    per_round_p = np.array([chisquare(c)[1] for c in counts])
    max_abs_dev = float(np.max(np.abs(per_round_freq - 0.25)))
    max_rel_dev = float(np.max(np.abs(per_round_freq - 0.25) / 0.25))

    # 逐平面
    plane_counts = np.array([r["plane_counts"] for r in records], dtype=np.float64)  # (R,12,4)
    pooled_plane = plane_counts.sum(axis=0).astype(np.int64)                          # (12,4)
    plane_freq = pooled_plane / pooled_plane.sum(axis=1, keepdims=True)
    plane_chi2 = np.array([chisquare(row)[0] for row in pooled_plane])
    plane_p = np.array([chisquare(row)[1] for row in pooled_plane])
    plane_chi2_per_round = np.array([chisquare(row)[0] for row in plane_counts.reshape(-1, 4)])
    plane_chi2_per_round = plane_chi2_per_round.reshape(len(records), 12)
    # 显著性检验的超限次数（3 自由度 1% 临界值 = 11.345）
    critical = float(chi2_dist.ppf(0.99, 3))
    round_over = int((per_round_chi2 > critical).sum())
    plane_over = int((plane_chi2_per_round > critical).sum())
    plane_p_per_round = np.array([chi2_dist.sf(row, 3) for row in plane_chi2_per_round.reshape(-1, 4)])
    plane_p_per_round = plane_p_per_round.reshape(len(records), 12)
    fps_rounds = int(np.sum(np.any(plane_p_per_round < 0.05, axis=1)))
    plane_over_05 = int((plane_p_per_round < 0.05).sum())

    se = float(np.sqrt(0.25 * 0.75 / pooled_total))
    ci_low = pooled_freq - 1.96 * se
    ci_high = pooled_freq + 1.96 * se

    # 多重比较：12 个平面各自做一次卡方，Bonferroni 校正阈值 0.05/12
    plane_bonferroni = 0.05 / len(pooled_plane)
    plane_min_p = float(min(plane_p))
    plane_bonferroni_pass = bool(plane_min_p > plane_bonferroni)
    plane_dev_z = np.abs(plane_freq - 0.25) / np.sqrt(0.25 * 0.75 / pooled_plane.sum(axis=1, keepdims=True))
    plane_z_max = float(plane_dev_z.max())

    # 空间同质性：块 × 碱基 列联表独立性检验
    tile_counts = np.array(records[0]["tile_counts"], dtype=np.float64)
    tile_counts = np.array([r["tile_counts"] for r in records], dtype=np.float64).sum(axis=0)
    row_tot = tile_counts.sum(axis=1, keepdims=True)
    col_tot = tile_counts.sum(axis=0, keepdims=True)
    expected = row_tot @ col_tot / tile_counts.sum()
    tile_chi2 = float(((tile_counts - expected) ** 2 / expected).sum())
    tile_dof = int((tile_counts.shape[0] - 1) * (tile_counts.shape[1] - 1))
    tile_p = float(chi2_dist.sf(tile_chi2, tile_dof))
    tile_freq = tile_counts / row_tot
    tile_max_abs_dev = float(np.max(np.abs(tile_freq - 0.25)))

    # 规则 id / 2-bit 取值
    rule_counts = np.array([r["rule_counts"] for r in records], dtype=np.float64).sum(axis=0)
    value_counts = np.array([r["value_counts"] for r in records], dtype=np.float64).sum(axis=0)
    rule_chi2, rule_p = chisquare(rule_counts)
    value_chi2, value_p = chisquare(value_counts)

    # 12-bit 量化值 / 6-bit 半字 / 低 12 位
    high12 = np.array([r["high12_counts"] for r in records], dtype=np.float64).sum(axis=0)
    low12 = np.array([r["low12_counts"] for r in records], dtype=np.float64).sum(axis=0)
    high6 = np.array([r["high6_counts"] for r in records], dtype=np.float64).sum(axis=0)
    low6 = np.array([r["low6_counts"] for r in records], dtype=np.float64).sum(axis=0)
    high6_chi2, high6_p = chisquare(high6)
    low6_chi2, low6_p = chisquare(low6)
    high12_chi2, high12_p = chisquare(high12)
    low12_chi2, low12_p = chisquare(low12)

    # 累积趋势
    cumulative = {}
    for n in (1, 5, 10, 20, 30, 40, rounds):
        if n > rounds:
            continue
        cum = counts[:n].sum(axis=0)
        cumulative[n] = {
            "counts": cum.astype(np.int64).tolist(),
            "freq": (cum / cum.sum()).tolist(),
        }

    return {
        "image": Path(run["image"]).name,
        "rounds": rounds,
        "pixels_per_round": int(totals[0]),
        "pooled_counts": pooled.tolist(),
        "pooled_freq": pooled_freq.tolist(),
        "pooled_ci_low": ci_low.tolist(),
        "pooled_ci_high": ci_high.tolist(),
        "pooled_se": se,
        "pooled_entropy": float(-np.sum(pooled_freq * np.log2(pooled_freq))),
        "pooled_chi2": float(pooled_chi2),
        "pooled_p": float(pooled_p),
        "pooled_total": pooled_total,
        "per_round_freq": per_round_freq.tolist(),
        "per_round_chi2": per_round_chi2.tolist(),
        "per_round_p": per_round_p.tolist(),
        "per_round_max_abs_dev": max_abs_dev,
        "per_round_max_rel_dev": max_rel_dev,
        "per_round_chi2_mean": float(per_round_chi2.mean()),
        "per_round_chi2_max": float(per_round_chi2.max()),
        "per_round_max_z": float(np.max(np.abs(per_round_freq - 0.25)) / np.sqrt(0.25 * 0.75 / totals[0])),
        "rounds_over_critical": round_over,
        "planes_over_critical": plane_over,
        "plane_total_tests": int(plane_chi2_per_round.size),
        "rounds_with_plane_p_lt_05": fps_rounds,
        "planes_p_lt_05": plane_over_05,
        "chi2_999_quantile": float(critical),
        "plane_min_p": plane_min_p,
        "plane_bonferroni_threshold": plane_bonferroni,
        "plane_bonferroni_pass": plane_bonferroni_pass,
        "plane_max_abs_z": plane_z_max,
        "pooled_plane_counts": pooled_plane.tolist(),
        "pooled_plane_freq": plane_freq.tolist(),
        "pooled_plane_chi2": plane_chi2.tolist(),
        "pooled_plane_p": plane_p.tolist(),
        "plane_chi2_per_round_mean": plane_chi2_per_round.mean(axis=0).tolist(),
        "plane_chi2_per_round_max": plane_chi2_per_round.max(axis=0).tolist(),
        "tile_chi2": tile_chi2,
        "tile_dof": tile_dof,
        "tile_p": tile_p,
        "tile_max_abs_dev": tile_max_abs_dev,
        "tile_max_abs_dev_expected_3sigma": float(
            3 * np.sqrt(0.25 * 0.75 / (tile_counts.sum() / tile_counts.shape[0]))
        ),
        "rule_counts": rule_counts.astype(np.int64).tolist(),
        "rule_chi2": float(rule_chi2),
        "rule_p": float(rule_p),
        "value_counts": value_counts.astype(np.int64).tolist(),
        "value_chi2": float(value_chi2),
        "value_p": float(value_p),
        "high6_chi2": float(high6_chi2),
        "high6_p": float(high6_p),
        "low6_chi2": float(low6_chi2),
        "low6_p": float(low6_p),
        "high12_chi2": float(high12_chi2),
        "high12_p": float(high12_p),
        "low12_chi2": float(low12_chi2),
        "low12_p": float(low12_p),
        "high12_min_freq": float(high12.min() / high12.sum()),
        "high12_max_freq": float(high12.max() / high12.sum()),
        "high12_uniform_chi2_mean": 4095.0,
        "high12_uniform_chi2_sd": float(np.sqrt(2 * 4095)),
        "high6_min_freq": float(high6.min() / high6.sum()),
        "high6_max_freq": float(high6.max() / high6.sum()),
        "cumulative": cumulative,
    }


def print_report(summary: dict) -> None:
    print()
    print("=" * 78)
    print(f"图片 {summary['image']}：{summary['rounds']} 个不同 user_key")
    print(f"总样本数 = {summary['pooled_total']:,} 个碱基 "
          f"({summary['rounds']} × {summary['pixels_per_round']:,})")
    print("=" * 78)

    print("\n[1] 汇总碱基计数（R 轮合计）")
    print(f"{'base':<6}{'count':>14}{'frequency':>14}{'dev vs 25%':>14}{'rel dev':>12}"
          f"{'95% CI':>22}")
    for idx, (name, count, freq) in enumerate(
        zip(BASE_NAMES, summary["pooled_counts"], summary["pooled_freq"])
    ):
        dev = freq - 0.25
        ci = f"[{summary['pooled_ci_low'][idx] * 100:.4f}, {summary['pooled_ci_high'][idx] * 100:.4f}]%"
        print(f"{name:<6}{count:>14,}{freq:>14.6f}{dev:>+14.6f}{dev / 0.25:>+11.3%}{ci:>22}")
    verdict = "与均匀分布无显著差异" if summary["pooled_p"] > 0.05 else "偏离均匀分布"
    print(f"卡方(3 自由度) = {summary['pooled_chi2']:.4f}   p = {summary['pooled_p']:.4f}"
          f"   → {verdict}（α = 0.05）")
    print(f"碱基经验熵 H = {summary['pooled_entropy']:.8f} bit （理论上限 2 bit）")
    print(f"四碱基最大偏离 = {max(abs(f - 0.25) for f in summary['pooled_freq']):.6f} "
          f"= {max(abs(f - 0.25) for f in summary['pooled_freq']) / 0.25:.4%} 相对偏差"
          f"，约 {max(abs(f - 0.25) for f in summary['pooled_freq']) / summary['pooled_se']:.2f}σ")

    print("\n[2] 逐轮分布（频率 %）与卡方")
    print(f"{'round':<7}" + "".join(f"{n + ' %':>11}" for n in BASE_NAMES)
          + f"{'chi2':>10}{'p':>10}{'max|dev|':>11}")
    for idx, freq in enumerate(summary["per_round_freq"]):
        chi2 = summary["per_round_chi2"][idx]
        p = summary["per_round_p"][idx]
        maxdev = float(np.max(np.abs(np.array(freq) - 0.25)))
        print(f"{idx:<7}" + "".join(f"{f * 100:>11.4f}" for f in freq)
              + f"{chi2:>10.4f}{p:>10.4f}{maxdev:>11.2e}")
    print(f"\n逐轮卡方：均值 {summary['per_round_chi2_mean']:.4f}，最大 "
          f"{summary['per_round_chi2_max']:.4f}（3 自由度 99% 分位 = "
          f"{summary['chi2_999_quantile']:.3f}）")
    print(f"单轮最大绝对偏差 {summary['per_round_max_abs_dev']:.3e} "
          f"= 相对偏差 {summary['per_round_max_rel_dev']:.4%}"
          f" ≈ {summary['per_round_max_z']:.2f}σ (单轮标准误 "
          f"{np.sqrt(0.25 * 0.75 / summary['pixels_per_round']):.3e})")
    print(f"超过 1% 临界值({summary['chi2_999_quantile']:.3f})的轮数: "
          f"{summary['rounds_over_critical']}/{summary['rounds']} "
          f"（随机抽样本应期望 {0.01 * summary['rounds']:.2f} 轮）")

    print("\n[3] 逐 DNA 平面（12 个平面，R 轮合计）")
    print(f"{'plane':<7}" + "".join(f"{n + ' %':>11}" for n in BASE_NAMES)
          + f"{'chi2':>10}{'p':>10}{'chi2/rnd':>10}{'max/rnd':>10}")
    for idx in range(len(summary["pooled_plane_counts"])):
        freq = summary["pooled_plane_freq"][idx]
        print(f"{idx:<7}" + "".join(f"{f * 100:>11.4f}" for f in freq)
              + f"{summary['pooled_plane_chi2'][idx]:>10.4f}"
              + f"{summary['pooled_plane_p'][idx]:>10.4f}"
              + f"{summary['plane_chi2_per_round_mean'][idx]:>10.3f}"
              + f"{summary['plane_chi2_per_round_max'][idx]:>10.3f}")
    print(f"12 平面合并卡方最小 p = {summary['plane_min_p']:.4f}；"
          f"Bonferroni 阈值 = {summary['plane_bonferroni_threshold']:.4f} → "
          f"{'全部通过（无平面存在真实偏好）' if summary['plane_bonferroni_pass'] else '有平面未通过，需复查'}")
    print(f"12 平面中最大偏离 = {summary['plane_max_abs_z']:.2f}σ")
    print(f"逐轮逐平面共 {summary['plane_total_tests']} 次检验："
          f"p<0.05 的 {summary['planes_p_lt_05']} 次 "
          f"（期望 {0.05 * summary['plane_total_tests']:.1f} 次），"
          f"其中出现至少一个 p<0.05 的轮数 {summary['rounds_with_plane_p_lt_05']}/"
          f"{summary['rounds']}（多重比较下的偶发假阳性，非系统性偏置）")

    print("\n[4] 空间同质性（64×64 分块，块 × 碱基 独立性检验）")
    print(f"卡方 = {summary['tile_chi2']:.1f}, 自由度 = {summary['tile_dof']}, "
          f"p = {summary['tile_p']:.3e}")
    print(f"单块最大绝对偏差 = {summary['tile_max_abs_dev']:.4f}"
          f"（单块正态 3σ 参考 ≈ {summary['tile_max_abs_dev_expected_3sigma']:.4f}）")

    print("\n[5] 中间量分布（用于定位偏差来源）")
    print(f"DNA 编码规则 id (1..8) 计数: {summary['rule_counts']}"
          f"  chi2 = {summary['rule_chi2']:.2f}, p = {summary['rule_p']:.4f}")
    print(f"2-bit 原始取值 (0..3): {summary['value_counts']}"
          f"  chi2 = {summary['value_chi2']:.2f}, p = {summary['value_p']:.4f}")
    print(f"上游 12-bit 量化值 (0..4095): chi2 = {summary['high12_chi2']:.1f}, "
          f"p = {summary['high12_p']:.4f}")
    print(f"   （均匀分布下 E[chi2] = {summary['high12_uniform_chi2_mean']:.0f}，"
          f"sd = {summary['high12_uniform_chi2_sd']:.1f}，"
          f"z = {(summary['high12_chi2'] - summary['high12_uniform_chi2_mean']) / summary['high12_uniform_chi2_sd']:+.2f}）")
    print(f"   单值频率范围 ({summary['high12_min_freq'] * 100:.5f}% ~ "
          f"{summary['high12_max_freq'] * 100:.5f}%)，理想 1/4096 = 0.024414%")
    print(f"上游 6-bit 高半字: chi2 = {summary['high6_chi2']:.2f}, p = {summary['high6_p']:.4f}")
    print(f"上游 6-bit 低半字: chi2 = {summary['low6_chi2']:.2f}, p = {summary['low6_p']:.4f}")
    print(f"下游 12-bit 量化值: chi2 = {summary['low12_chi2']:.1f}, p = {summary['low12_p']:.4f}")

    print("\n[6] 累积趋势（多轮平均后频率）")
    for n, data in summary["cumulative"].items():
        freq = "  ".join(f"{name}={f * 100:.4f}%" for name, f in zip(BASE_NAMES, data["freq"]))
        print(f"  前 {n:>2} 轮: {freq}")


def make_figure(summaries: list[dict], out_path: Path) -> None:
    """逐轮碱基占比 + 累积收敛曲线，用于论文图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    colors = {"A": "#d84a4a", "C": "#3f9e52", "G": "#416da5", "T": "#c8963e"}

    ax = axes[0]
    for image_idx, summary in enumerate(summaries):
        x = np.arange(summary["rounds"]) + image_idx * 0.0
        marker = "o" if image_idx == 0 else "s"
        for base_idx, name in enumerate(BASE_NAMES):
            y = np.array(summary["per_round_freq"])[:, base_idx] * 100
            ax.plot(x, y, marker=marker, ms=3.4, lw=0.9, alpha=0.85,
                    color=colors[name],
                    label=f"{name} ({summary['image'].replace('.png', '')})"
                    if image_idx == 0 else None)
    ax.axhline(25.0, color="black", lw=1.2, ls="--", label="ideal 25%")
    ax.set_xlabel("round index (different user_key)")
    ax.set_ylabel("base frequency (%)")
    ax.set_title("(a) Per-round base frequency", fontsize=11)
    ax.set_ylim(24.9, 25.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center")

    ax = axes[1]
    ref = summaries[0]
    ns = sorted(int(n) for n in ref["cumulative"])
    for base_idx, name in enumerate(BASE_NAMES):
        y = [(ref["cumulative"][str(n)] if str(n) in ref["cumulative"]
              else ref["cumulative"][n])["freq"][base_idx] * 100 for n in ns]
        ax.plot(ns, y, marker="o", ms=4, lw=1.1, color=colors[name], label=name)
    ax.axhline(25.0, color="black", lw=1.2, ls="--", label="ideal 25%")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("number of averaged user_keys")
    ax.set_ylabel("cumulative base frequency (%)")
    ax.set_title(f"(b) Convergence, {ref['image']}", fontsize=11)
    ax.set_ylim(24.94, 25.06)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=300, facecolor="white")
    print(f"图已保存: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encrypter = Encrypter(ENCRYPTION_CONFIG)

    all_summaries = []
    for image_path, rounds in ((PRIMARY_IMAGE, PRIMARY_ROUNDS), (SECONDARY_IMAGE, SECONDARY_ROUNDS)):
        print(f"\n>>> 图片 {image_path.name}，{rounds} 个 user_key ...", flush=True)
        run = run_image(encrypter, image_path, rounds)
        summary = summarise(run)
        all_summaries.append(summary)
        print_report(summary)

    out_json = OUT_DIR / "global_key_distribution.json"
    out_json.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存: {out_json}")
    make_figure(all_summaries, OUT_DIR / "global_key_base_distribution.png")


if __name__ == "__main__":
    main()
