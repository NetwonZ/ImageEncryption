"""抗差分攻击性能评估：计算并展示 NPCR 与 UACI 指标。

测试方法（标准差分分析）：
    对每张明文图像 P，随机选取一个像素并将其灰度值改变 1（±1）得到 P'；
    用同一密钥分别加密 P 和 P' 得到密文 C 与 C'，然后计算：

    NPCR = mean(C != C') * 100%                （像素变化率）
    UACI = mean(|C - C'| / 255) * 100%         （平均变化强度）

理论理想值：NPCR ≈ 99.6094%，UACI ≈ 33.4635%。

注意：本系统的混沌密钥序列只由图像尺寸与系统参数决定（与明文无关），
因此两次加密天然使用同一密钥，满足差分分析的对照条件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # 支持直接运行与包模块运行。
    from .image_io import load_grayscale
    from .pipeline import ImageCryptosystem
except ImportError:  # pragma: no cover - 直接脚本执行时使用
    from image_io import load_grayscale
    from pipeline import ImageCryptosystem

# 8-bit 灰度图的理论理想值（NPCR 为 (1 - 1/256) * 100）
IDEAL_NPCR = 99.6094
IDEAL_UACI = 33.4635
BASE_DIR = Path(__file__).resolve().parent


def compute_npcr_uaci(cipher_a: np.ndarray, cipher_b: np.ndarray) -> tuple[float, float]:
    """计算两幅同尺寸密文图像之间的 NPCR 与 UACI（百分比）。"""
    a = np.asarray(cipher_a)
    b = np.asarray(cipher_b)
    if a.shape != b.shape:
        raise ValueError(f"两幅密文尺寸不一致: {a.shape} vs {b.shape}")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    npcr = float(np.mean(a != b) * 100.0)
    uaci = float(np.mean(np.abs(a - b)) / 255.0 * 100.0)
    return npcr, uaci


def _one_pixel_variant(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """复制图像并随机修改一个像素的灰度值（±1），返回副本及像素坐标。"""
    variant = image.copy()
    row = int(rng.integers(0, image.shape[0]))
    col = int(rng.integers(0, image.shape[1]))
    value = int(variant[row, col])
    variant[row, col] = value - 1 if value == 255 else value + 1
    return variant, (row, col)


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """以等宽 ASCII 表格打印结果。"""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _format_row(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, widths)) + " |"

    print(separator)
    print(_format_row(headers))
    print(separator)
    for row in rows:
        print(_format_row(row))
    print(separator)


def test_differential_attack(
    image_paths: list[str | Path],
    *,
    dt: float = 0.01,
    pre_iterations: int = 1000,
    seed: int = 20260721,
    diffusion_mode: str = "hardened",
) -> list[dict]:
    """评估加密系统的抗差分性能（NPCR / UACI），并将结果打印为表格。

    参数:
        image_paths: n 张输入灰度图像的路径列表（算法要求正方形图像）。
        dt: Euler 积分步长，与 ImageCryptosystem 一致。
        pre_iterations: 混沌序列预迭代次数，与 ImageCryptosystem 一致。
        seed: 随机种子，控制"修改哪个像素"，保证实验可复现。

    返回:
        每张图像的结果字典列表，含 npcr_percent / uaci_percent 等字段。
    """
    if not image_paths:
        raise ValueError("image_paths 不能为空，至少需要一张图像")

    rng = np.random.default_rng(seed)
    system = ImageCryptosystem(
        dt=dt,
        pre_iterations=pre_iterations,
        diffusion_mode=diffusion_mode,
    )
    results: list[dict] = []

    for path in image_paths:
        path = Path(path)
        plain = load_grayscale(path)
        if plain.shape[0] != plain.shape[1]:
            raise ValueError(f"算法要求正方形图像，{path.name} 的尺寸为 {plain.shape}")

        modified, (row, col) = _one_pixel_variant(plain, rng)
        cipher_a = system.encrypt(plain).ciphertext
        cipher_b = system.encrypt(modified).ciphertext
        npcr, uaci = compute_npcr_uaci(cipher_a, cipher_b)

        results.append(
            {
                "image": str(path),
                "shape": tuple(plain.shape),
                "changed_pixel": (row, col),
                "npcr_percent": npcr,
                "uaci_percent": uaci,
                "diffusion_mode": diffusion_mode,
            }
        )

    headers = ("Image", "Size", "Changed Pixel", "NPCR (%)", "UACI (%)")
    rows = [
        (
            Path(item["image"]).name,
            f"{item['shape'][0]}x{item['shape'][1]}",
            str(item["changed_pixel"]),
            f"{item['npcr_percent']:.4f}",
            f"{item['uaci_percent']:.4f}",
        )
        for item in results
    ]

    avg_npcr = float(np.mean([item["npcr_percent"] for item in results]))
    avg_uaci = float(np.mean([item["uaci_percent"] for item in results]))
    rows.append(("Average", "-", "-", f"{avg_npcr:.4f}", f"{avg_uaci:.4f}"))
    rows.append(("Ideal", "-", "-", f"{IDEAL_NPCR:.4f}", f"{IDEAL_UACI:.4f}"))

    print(f"=== 抗差分攻击性能评估 (NPCR / UACI, mode={diffusion_mode}) ===")
    _print_table(headers, rows)
    return results


if __name__ == "__main__":
    demo_images = [
        BASE_DIR / "images" / "img3.png",
        BASE_DIR / "images" / "img4.png",
        BASE_DIR / "images" / "img5.png",
        BASE_DIR / "images" / "img6.png",
        BASE_DIR / "images" / "img7.png",
        BASE_DIR / "output" / "demo" / "img3.png",
    ]
    available = [p for p in demo_images if p.exists()]
    if not available:
        raise SystemExit("未找到测试图像：请调用test_differential_attack并传入至少一张正方形灰度图。")
    test_differential_attack(available)
