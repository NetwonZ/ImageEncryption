"""运行论文图像加密系统的入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from .image_io import create_demo_image, load_grayscale, save_grayscale
from .pipeline import ImageCryptosystem, debug_summary



@dataclass
class Config:
    # 输入灰度图像路径；如果设为 None 或文件不存在，则自动生成演示图
    input_path: Path | None = Path(r"C:\ImageEncryption\papers\adaptive_scrambling\output\demo\img3.png")
    # 输出结果保存目录
    output_dir: Path = Path("papers/adaptive_scrambling/output/demo")
    # 演示图边长（仅在 input_path 为 None 时生效）
    size: int = 256
    # Euler 步长 Δτ（论文未公开，默认 0.01）
    dt: float = 0.01
    # 预迭代次数（论文未公开）
    pre_iterations: int = 1000


def main() -> None:
    # 实例化配置类
    cfg = Config()

    st = time.time()

    # 1. 加载或生成图像
    if cfg.input_path and cfg.input_path.exists():
        image = load_grayscale(cfg.input_path)
    else:
        print(f"ℹ️ 未在路径找到输入图像 ({cfg.input_path})，将自动生成 {cfg.size}x{cfg.size} 测试演示图...")
        image = create_demo_image(cfg.size)

    # 2. 校验图像维度
    if image.shape[0] != image.shape[1]:
        raise ValueError("论文 Algorithm 2 使用正方形块，输入图像必须是正方形")

    # 3. 初始化并运行加密系统
    system = ImageCryptosystem(dt=cfg.dt, pre_iterations=cfg.pre_iterations)
    result = system.encrypt(image)

    et = time.time()
    print(f"=== 加密流程完成，耗时 {et - st:.6f} 秒 ===")

    # 4. 保存加密中间产物及结果图像
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    save_grayscale(image, out / "01_original.png")
    save_grayscale(result.scrambled, out / "02_scrambled.png")
    save_grayscale(result.local_diffusion, out / "03_diffusion_local_xor.png")
    save_grayscale(result.ciphertext, out / "04_ciphertext.png")
    save_grayscale(result.decrypted, out / "05_decrypted.png")

    # 5. 汇总并打印指标
    summary = debug_summary(result)
    summary["round_trip_equal_to_original"] = bool(np.array_equal(result.decrypted, image))
    summary["input"] = str(cfg.input_path) if (cfg.input_path and cfg.input_path.exists()) else "generated_demo"

    print("=== 加密流程指标汇总 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()