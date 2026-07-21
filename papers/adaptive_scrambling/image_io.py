"""灰度图像读取、写出和无输入时的可重复演示图。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def load_grayscale(path: str | Path) -> np.ndarray:
    """读取并转换为uint8灰度数组。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_grayscale(array: np.ndarray, path: str | Path) -> None:
    """保存二维uint8数组为PNG。"""
    data = np.asarray(array, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("只能保存二维灰度数组")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data, mode="L").save(target)


def create_demo_image(size: int = 256) -> np.ndarray:
    """生成无需外部数据集的医学风格灰度演示图。"""
    y, x = np.mgrid[0:size, 0:size]
    center = size / 2.0
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    image = 35 + 95 * np.exp(-(r / (size * 0.38)) ** 2)
    image += 42 * np.exp(-(((x - center * 0.72) / (size * 0.15)) ** 2
                           + ((y - center * 0.78) / (size * 0.25)) ** 2))
    image -= 35 * np.exp(-(((x - center * 1.28) / (size * 0.15)) ** 2
                            + ((y - center * 0.78) / (size * 0.25)) ** 2))
    image += 12 * np.sin(x / 7.0) * np.cos(y / 11.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image, mode="L")
    draw = ImageDraw.Draw(pil)
    draw.ellipse((size * .20, size * .22, size * .47, size * .79), outline=175, width=max(1, size // 128))
    draw.ellipse((size * .53, size * .22, size * .80, size * .79), outline=175, width=max(1, size // 128))
    return np.asarray(pil, dtype=np.uint8)
