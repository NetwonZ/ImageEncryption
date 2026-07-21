"""论文Algorithm 2：自适应分块、块内置乱和块间置乱。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ScrambleRoundConfig:
    """一轮自适应置乱的最大、最小块边长。"""

    max_block_size: int = 32
    min_block_size: int = 8

    def __post_init__(self) -> None:
        if self.max_block_size < 2 or self.min_block_size < 1:
            raise ValueError("块尺寸必须为正数且最大块至少为2")
        if self.min_block_size > self.max_block_size:
            raise ValueError("min_block_size不能大于max_block_size")


@dataclass
class BlockRecord:
    """保存一个最终块的源位置、目标位置和块内变换。"""

    source_row: int
    source_col: int
    dest_row: int
    dest_col: int
    height: int
    width: int
    transform: str
    source_tile: int
    dest_tile: int


@dataclass
class ScrambleKey:
    """一轮置乱的全部逆置乱信息。"""

    image_shape: tuple[int, int]
    config: ScrambleRoundConfig
    threshold: float
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    blocks: list[BlockRecord]
    seed: int


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """计算Gx、Gy并返回sqrt(Gx^2+Gy^2)。"""
    image_f = np.asarray(image, dtype=np.float64)
    if image_f.size == 1:
        return np.zeros_like(image_f)
    gx, gy = np.gradient(image_f)
    return np.sqrt(gx * gx + gy * gy)


def block_gradient(block: np.ndarray) -> float:
    """论文公式(2)：G(Q)=1/h^2 * sum sqrt(Gx(i,j)^2+Gy(i,j)^2)。"""
    h = max(block.shape)
    if h == 0:
        return 0.0
    return float(np.sum(_gradient_magnitude(block)) / (h * h))


def _zigzag_indices(height: int, width: int) -> list[tuple[int, int]]:
    indices: list[tuple[int, int]] = []
    for diagonal in range(height + width - 1):
        r0 = max(0, diagonal - (width - 1))
        r1 = min(height - 1, diagonal)
        rows = range(r1, r0 - 1, -1) if diagonal % 2 == 0 else range(r0, r1 + 1)
        for row in rows:
            indices.append((row, diagonal - row))
    return indices


def _apply_transform(block: np.ndarray, transform: str) -> np.ndarray:
    if transform == "zigzag":
        order = _zigzag_indices(*block.shape)
        values = np.asarray([block[r, c] for r, c in order], dtype=block.dtype)
        return values.reshape(block.shape)
    if transform.startswith("rot"):
        angle = int(transform[3:])
        return np.rot90(block, k=angle // 90).copy()
    raise ValueError(f"未知块内变换: {transform}")


def _inverse_transform(block: np.ndarray, transform: str) -> np.ndarray:
    if transform == "zigzag":
        order = _zigzag_indices(*block.shape)
        result = np.empty_like(block)
        for idx, (r, c) in enumerate(order):
            result[r, c] = block.flat[idx]
        return result
    if transform.startswith("rot"):
        angle = int(transform[3:])
        return np.rot90(block, k=(-angle // 90) % 4).copy()
    raise ValueError(f"未知块内变换: {transform}")


def _split_tile(
    image: np.ndarray,
    row: int,
    col: int,
    h: int,
    min_size: int,
    threshold: float,
    tile_id: int,
    records: list[tuple[int, int, int, int, int]],
) -> None:
    """递归四分裂；记录(row,col,h,w,tile_id)。"""
    block = image[row:row + h, col:col + h]
    if h <= min_size or h < 2:
        records.append((row, col, h, h, tile_id))
        return
    if block_gradient(block) <= threshold:
        records.append((row, col, h, h, tile_id))
        return
    half = h // 2
    if half < 1:
        records.append((row, col, h, h, tile_id))
        return
    _split_tile(image, row, col, half, min_size, threshold, tile_id, records)
    _split_tile(image, row, col + half, half, min_size, threshold, tile_id, records)
    _split_tile(image, row + half, col, half, min_size, threshold, tile_id, records)
    _split_tile(image, row + half, col + half, half, min_size, threshold, tile_id, records)


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("论文算法针对二维灰度图像，输入必须是H×W数组")
    if array.shape[0] != array.shape[1]:
        raise ValueError("论文的四分裂块为正方形，当前实现要求输入为正方形图像")
    if array.shape[0] < 2:
        raise ValueError("图像边长必须至少为2")
    if array.dtype != np.uint8:
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("图像像素必须位于[0,255]")
        array = array.astype(np.uint8)
    return array


def adaptive_scramble(
    image: np.ndarray,
    *,
    seed: int,
    config: ScrambleRoundConfig = ScrambleRoundConfig(),
) -> tuple[np.ndarray, ScrambleKey]:
    """执行一轮自适应置乱并返回置乱图像及可逆密钥。"""
    image = _validate_image(image)
    h, w = image.shape
    B = config.max_block_size
    if h % B != 0 or w % B != 0:
        raise ValueError(
            f"图像尺寸{image.shape}必须能被max_block_size={B}整除，"
            "以保持论文Algorithm 2的宏块排列定义。"
        )
    tile_rows, tile_cols = h // B, w // B
    macro_count = tile_rows * tile_cols
    threshold = 1.5 * float(np.mean(_gradient_magnitude(image)))
    rng = np.random.default_rng(int(seed) % (2 ** 32))
    raw_records: list[tuple[int, int, int, int, int]] = []

    for tile_id in range(macro_count):
        tile_row, tile_col = divmod(tile_id, tile_cols)
        _split_tile(image, tile_row * B, tile_col * B, B,
                    config.min_block_size, threshold, tile_id, raw_records)

    permutation = rng.permutation(macro_count).astype(np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(macro_count, dtype=np.int64)
    output = np.empty_like(image)
    records: list[BlockRecord] = []
    transforms = ("zigzag", "rot0", "rot90", "rot180", "rot270")

    for source_row, source_col, bh, bw, source_tile in raw_records:
        transform = transforms[int(rng.integers(0, len(transforms)))]
        transformed = _apply_transform(image[source_row:source_row + bh,
                                               source_col:source_col + bw], transform)
        src_tile_row, src_tile_col = divmod(source_tile, tile_cols)
        local_row = source_row - src_tile_row * B
        local_col = source_col - src_tile_col * B
        dest_tile = int(permutation[source_tile])
        dst_tile_row, dst_tile_col = divmod(dest_tile, tile_cols)
        dest_row = dst_tile_row * B + local_row
        dest_col = dst_tile_col * B + local_col
        output[dest_row:dest_row + bh, dest_col:dest_col + bw] = transformed
        records.append(BlockRecord(source_row, source_col, dest_row, dest_col,
                                   bh, bw, transform, source_tile, dest_tile))

    key = ScrambleKey((h, w), config, threshold, permutation, inverse,
                      records, int(seed) % (2 ** 32))
    return output, key


def adaptive_unscramble(image: np.ndarray, key: ScrambleKey) -> np.ndarray:
    """使用ScrambleKey严格执行一轮逆置乱。"""
    image = _validate_image(image)
    if image.shape != key.image_shape:
        raise ValueError("逆置乱图像尺寸与置乱密钥不一致")
    output = np.empty_like(image)
    for block in key.blocks:
        data = image[block.dest_row:block.dest_row + block.height,
                     block.dest_col:block.dest_col + block.width]
        restored = _inverse_transform(data, block.transform)
        output[block.source_row:block.source_row + block.height,
               block.source_col:block.source_col + block.width] = restored
    return output


def choose_round_configs(size: int) -> tuple[ScrambleRoundConfig, ScrambleRoundConfig]:
    """为论文未公开的Bmax/Bmin提供可重复的两轮默认配置。"""
    if size < 2:
        raise ValueError("图像边长太小")
    max_size = 1 << (int(size).bit_length() - 1)
    max_size = min(32, max_size)
    if size % max_size != 0:
        # 对论文中常见的512/1024以及测试图像，max_size总能整除；
        # 这里退到一个确实可整除的二次幂，避免改变像素尺寸。
        divisors = [2 ** p for p in range(1, 6) if size % (2 ** p) == 0]
        if not divisors:
            raise ValueError("图像边长需要包含2的因子，以支持论文的正方形四分裂")
        max_size = min(32, max(divisors))
    first_min = max(1, max_size // 4)
    second_max = max(2, max_size // 2)
    second_min = max(1, second_max // 4)
    return (ScrambleRoundConfig(max_size, first_min),
            ScrambleRoundConfig(second_max, second_min))


def scramble_rounds(
    image: np.ndarray,
    *,
    seeds: tuple[int, int],
    configs: tuple[ScrambleRoundConfig, ScrambleRoundConfig] | None = None,
) -> tuple[np.ndarray, list[ScrambleKey]]:
    """执行论文所述的两轮adaptive scrambling。"""
    image = _validate_image(image)
    configs = configs or choose_round_configs(image.shape[0])
    current = image
    keys: list[ScrambleKey] = []
    for seed, config in zip(seeds, configs):
        current, key = adaptive_scramble(current, seed=seed, config=config)
        keys.append(key)
    return current, keys


def unscramble_rounds(image: np.ndarray, keys: list[ScrambleKey]) -> np.ndarray:
    """按两轮置乱的相反顺序执行逆置乱。"""
    current = _validate_image(image)
    for key in reversed(keys):
        current = adaptive_unscramble(current, key)
    return current


def key_summary(keys: list[ScrambleKey]) -> list[dict[str, Any]]:
    """生成适合调试输出的置乱中间变量摘要。"""
    return [
        {
            "shape": key.image_shape,
            "max_block": key.config.max_block_size,
            "min_block": key.config.min_block_size,
            "gradient_threshold": key.threshold,
            "block_count": len(key.blocks),
            "permutation_head": key.permutation[: min(10, len(key.permutation))].tolist(),
            "seed": key.seed,
        }
        for key in keys
    ]
