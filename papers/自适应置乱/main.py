"""运行论文图像加密系统的入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from image_io import create_demo_image, load_grayscale, save_grayscale
from pipeline import ImageCryptosystem, debug_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="忆阻混沌神经元图像加密复现")
    parser.add_argument("--input", type=Path,default=Path(r"D:\papers\自适应置乱\output\demo\img3.png"),help="输入灰度图像；省略时生成演示图")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录")
    parser.add_argument("--size", type=int, default=256, help="演示图边长")
    parser.add_argument("--dt", type=float, default=0.01, help="Euler步长Δτ（论文未公开，默认0.01）")
    parser.add_argument("--pre-iterations", type=int, default=1000, help="预迭代次数（论文未公开）")
    return parser.parse_args()


def main() -> None:
    import time
    st = time.time()
    args = parse_args()
    image = load_grayscale(args.input) if args.input else create_demo_image(args.size)
    if image.shape[0] != image.shape[1]:
        raise ValueError("论文Algorithm 2使用正方形块，输入图像必须是正方形")
    system = ImageCryptosystem(dt=args.dt, pre_iterations=args.pre_iterations)
    result = system.encrypt(image)
    out = args.output_dir
    et = time.time()
    print(f"=== 加密流程完成，耗时 {et - st:.6f} 秒 ===")
    out.mkdir(parents=True, exist_ok=True)
    save_grayscale(image, out / "01_original.png")
    save_grayscale(result.scrambled, out / "02_scrambled.png")
    save_grayscale(result.local_diffusion, out / "03_diffusion_local_xor.png")
    save_grayscale(result.ciphertext, out / "04_ciphertext.png")
    save_grayscale(result.decrypted, out / "05_decrypted.png")

    summary = debug_summary(result)
    summary["round_trip_equal_to_original"] = bool(np.array_equal(result.decrypted, image))
    summary["input"] = str(args.input) if args.input else "generated_demo"
    print("=== 加密流程完成 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()
