from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

import PIL.Image as pil_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import time
from salomon import SalomoncouplingCML

@dataclass(frozen=True)
class ImageBlock:
    """A final block produced by adaptive image partitioning."""

    row: int
    col: int
    height: int
    width: int
    mean_gradient: float
    transform_id: int = 0

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.width)


@dataclass(frozen=True)
class AdaptivePartitionResult:
    """Partition output and metadata for later scrambling steps."""

    source_image: np.ndarray
    gray_image: np.ndarray
    gradient_method: str
    image_shape: tuple[int, ...]
    gray_shape: tuple[int, int]
    b_max: int
    b_min: int
    threshold: float
    global_mean_gradient: float
    blocks: tuple[ImageBlock, ...]


@dataclass(frozen=True)
class BlockPermutationIndices:
    """Per-block permutation indices for DNA block encryption."""

    plane_perm: np.ndarray
    row_perm: np.ndarray
    col_perm: np.ndarray


@dataclass(frozen=False)
class DNAEncodingResult:
    """Block-wise DNA encoding output for an RGB image."""

    bitplanes: np.ndarray
    encoded_dna_matrix: np.ndarray
    dna_matrix: np.ndarray
    block_rule_ids: np.ndarray
    pair_indices: tuple[tuple[int, int], ...]
    blocks: tuple[ImageBlock, ...]
    storage_format: str


@dataclass(frozen=True)
class BlockShuffleGroup:
    """One same-sized block group and its reversible inter-block mapping."""

    block_shape: tuple[int, int]
    target_block_indices: np.ndarray
    source_block_indices: np.ndarray


@dataclass(frozen=True)
class BlockShuffleResult:
    """Inter-block permutation result that remains decodable."""

    dna_matrix: np.ndarray
    block_rule_ids: np.ndarray
    bitplanes: np.ndarray
    groups: tuple[BlockShuffleGroup, ...]
    storage_format: str


@dataclass(frozen=True)
class BlockDNADiffusionResult:
    """Intra-block DNA diffusion result and per-block diffusion metadata."""

    dna_matrix: np.ndarray
    block_operation_ids: np.ndarray
    block_column_permutations: tuple[np.ndarray, ...]
    block_key_columns: tuple[np.ndarray, ...]
    storage_format: str


@dataclass(frozen=True)
class BlockDNADiffusionV2Result:
    """Simplified intra-block DNA diffusion result."""

    dna_matrix: np.ndarray
    block_key_matrices: tuple[np.ndarray, ...]
    channel_mode: str
    operation_name: str
    storage_format: str


@dataclass(frozen=True)
class GlobalDNADiffusionResult:
    """Global DNA diffusion result and metadata."""

    dna_matrix: np.ndarray
    permutation_indices: np.ndarray
    key_matrix: np.ndarray
    key_rule_id: int
    scheme: str
    parallel_mode: str
    parallel_size: int
    operation_name: str
    storage_format: str


DNA_BASE_TO_CODE = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}
DNA_CODE_TO_BASE = np.array(["A", "C", "G", "T"], dtype="<U1")
DNA_BASE_ORDER = ("A", "C", "G", "T")
DNA_SYMBOL_DTYPE = "<U1"
DNA_OPERATION_ID_TO_NAME = {
    1: "add",
    2: "sub",
    3: "xor",
}
DNA_OPERATION_CODE_TABLES = {
    "add": np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ],
        dtype=np.uint8,
    ),
    "sub": np.array(
        [
            [0, 3, 2, 1],
            [1, 0, 3, 2],
            [2, 1, 0, 3],
            [3, 2, 1, 0],
        ],
        dtype=np.uint8,
    ),
    "xor": np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.uint8,
    ),
}

# Rules are indexed by 2-bit values 00, 01, 10, 11 and stored as A/C/G/T -> 0/1/2/3.
# The numbering follows the user-provided table exactly.
DNA_RULE_TABLE = np.array(
    [
        [0, 2, 1, 3],  # Rule 1: A G C T
        [0, 1, 2, 3],  # Rule 2: A C G T
        [2, 0, 3, 1],  # Rule 3: G A T C
        [2, 3, 0, 1],  # Rule 4: G T A C
        [1, 0, 3, 2],  # Rule 5: C A T G
        [1, 3, 0, 2],  # Rule 6: C T A G
        [3, 2, 1, 0],  # Rule 7: T G C A
        [3, 1, 2, 0],  # Rule 8: T C G A
    ],
    dtype=np.uint8,
)
DNA_RULE_TABLE_INV = np.argsort(DNA_RULE_TABLE, axis=1).astype(np.uint8)

DNA_HIGH6_PAIR_INDICES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (8, 9),
    (10, 11),
    (12, 13),
    (16, 17),
    (18, 19),
    (20, 21),
)
LOW2_BITPLANE_INDICES = (6, 7, 14, 15, 22, 23)
ENCODED_HIGH6_BITPLANE_INDICES = tuple(sorted({index for pair in DNA_HIGH6_PAIR_INDICES for index in pair}))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a float grayscale array in the original value range."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.ndim != 3:
        raise ValueError("image must be a 2D grayscale array or a 3D color array")

    channels = image.shape[2]
    image = image.astype(np.float64, copy=False)
    if channels == 1:
        return image[..., 0]
    if channels >= 3:
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    raise ValueError("color image must have at least 3 channels")


def image_to_bitplanes(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image into 24 bitplanes ordered as R7..R0, G7..G0, B7..B0."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array with shape (H, W, 3)")

    image = image.astype(np.uint8, copy=False)
    height, width, _ = image.shape
    bitplanes = np.empty((24, height, width), dtype=np.uint8)
    plane_idx = 0
    for channel in range(3):
        for bit in range(7, -1, -1):
            bitplanes[plane_idx] = (image[:, :, channel] >> bit) & 1
            plane_idx += 1
    return bitplanes



def bitplanes_to_image(bitplanes: np.ndarray) -> np.ndarray:
    """Reconstruct an RGB image from 24 bitplanes ordered as R7..R0, G7..G0, B7..B0."""
    bitplanes = np.asarray(bitplanes, dtype=np.uint8)
    if bitplanes.shape[0] != 24:
        raise ValueError("bitplanes must have shape (24, H, W)")

    height, width = bitplanes.shape[1:]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    plane_idx = 0
    for channel in range(3):
        channel_data = np.zeros((height, width), dtype=np.uint8)
        for bit in range(7, -1, -1):
            channel_data |= (bitplanes[plane_idx] & 1) << bit
            plane_idx += 1
        image[:, :, channel] = channel_data
    return image


def compute_gradient_magnitude(gray_image: np.ndarray) -> np.ndarray:
    """Compute per-pixel gradient magnitude using numpy finite differences."""
    gray = np.asarray(gray_image, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError("gray_image must be a 2D array")
    if gray.size == 0:
        raise ValueError("gray_image must not be empty")

    grad_row, grad_col = np.gradient(gray)
    return np.hypot(grad_row, grad_col)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    threshold = 0.04045
    return np.where(rgb <= threshold, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8/float image to CIE Lab."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Lab conversion requires an RGB image with at least 3 channels")

    rgb = image[..., :3].astype(np.float64, copy=False)
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = _srgb_to_linear(rgb)

    xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = np.tensordot(rgb, xyz_matrix.T, axes=1)

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz_scaled = xyz / white
    delta = 6.0 / 29.0
    linear_mask = xyz_scaled > delta**3
    f_xyz = np.where(linear_mask, np.cbrt(xyz_scaled), xyz_scaled / (3 * delta**2) + 4.0 / 29.0)

    lab = np.empty_like(f_xyz)
    lab[..., 0] = 116.0 * f_xyz[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return lab


def compute_lab_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude after converting RGB to CIE Lab."""
    lab = rgb_to_lab(image)
    grad_sq_sum = np.zeros(lab.shape[:2], dtype=np.float64)
    for channel in range(3):
        grad_row, grad_col = np.gradient(lab[..., channel])
        grad_sq_sum += grad_row * grad_row + grad_col * grad_col
    return np.sqrt(grad_sq_sum)


def compute_di_zenzo_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute Di Zenzo vector gradient magnitude for a color image."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Di Zenzo gradient requires an RGB image with at least 3 channels")

    rgb = image[..., :3].astype(np.float64, copy=False)
    g_xx = np.zeros(rgb.shape[:2], dtype=np.float64)
    g_yy = np.zeros(rgb.shape[:2], dtype=np.float64)
    g_xy = np.zeros(rgb.shape[:2], dtype=np.float64)

    for channel in range(3):
        grad_row, grad_col = np.gradient(rgb[..., channel])
        g_xx += grad_col * grad_col
        g_yy += grad_row * grad_row
        g_xy += grad_col * grad_row

    trace = g_xx + g_yy
    diff = g_xx - g_yy
    lambda_max = 0.5 * (trace + np.sqrt(diff * diff + 4.0 * g_xy * g_xy))
    return np.sqrt(np.maximum(lambda_max, 0.0))


def compute_color_gradient_magnitude(image: np.ndarray, method: str = "gray") -> tuple[np.ndarray, np.ndarray, str]:
    """Compute gradient magnitude using the requested method.

    Returns ``(gray_image, gradient_magnitude, normalized_method_name)``.
    """
    normalized_method = method.lower().strip()
    image = np.asarray(image)
    gray = to_grayscale(image)

    if normalized_method == "gray":
        grad_mag = compute_gradient_magnitude(gray)
    elif normalized_method == "lab":
        grad_mag = compute_lab_gradient_magnitude(image)
    elif normalized_method in {"di_zenzo", "dizenzo", "di-zenzo"}:
        grad_mag = compute_di_zenzo_gradient_magnitude(image)
        normalized_method = "di_zenzo"
    else:
        raise ValueError("gradient_method must be one of: 'gray', 'lab', 'di_zenzo'")

    return gray, grad_mag, normalized_method


def adaptive_partition(
    image: np.ndarray,
    b_max: int,
    b_min: int,
    gradient_threshold: float | None = None,
    gradient_method: str = "gray",
) -> AdaptivePartitionResult:
    """Partition an image into adaptive blocks based on local gradient strength."""
    if not isinstance(b_max, int) or not isinstance(b_min, int):
        raise TypeError("b_max and b_min must be integers")
    if b_max <= 0 or b_min <= 0:
        raise ValueError("b_max and b_min must be positive")
    if b_min > b_max:
        raise ValueError("b_min must not be larger than b_max")

    source_image = np.asarray(image)
    gray, grad_mag, gradient_method = compute_color_gradient_magnitude(source_image, gradient_method)
    global_mean_gradient = float(np.mean(grad_mag))
    if gradient_threshold is None:
        threshold = 1.5 * global_mean_gradient
    else:
        threshold = float(gradient_threshold)
        if not np.isfinite(threshold):
            raise ValueError("gradient_threshold must be finite")

    height, width = gray.shape
    blocks: list[ImageBlock] = []

    def block_mean_gradient(row: int, col: int, block_h: int, block_w: int) -> float:
        region = grad_mag[row:row + block_h, col:col + block_w]
        return float(np.mean(region))

    def split_block(row: int, col: int, block_h: int, block_w: int) -> None:
        mean_grad = block_mean_gradient(row, col, block_h, block_w)

        can_split = block_h > 1 and block_w > 1
        reached_min = block_h <= b_min or block_w <= b_min
        if reached_min or mean_grad <= threshold or not can_split:
            blocks.append(
                ImageBlock(
                    row=row,
                    col=col,
                    height=block_h,
                    width=block_w,
                    mean_gradient=mean_grad,
                )
            )
            return

        half_h = block_h // 2
        half_w = block_w // 2
        if half_h == 0 or half_w == 0:
            blocks.append(
                ImageBlock(
                    row=row,
                    col=col,
                    height=block_h,
                    width=block_w,
                    mean_gradient=mean_grad,
                )
            )
            return

        sub_blocks = (
            (row, col, half_h, half_w),
            (row, col + half_w, half_h, block_w - half_w),
            (row + half_h, col, block_h - half_h, half_w),
            (row + half_h, col + half_w, block_h - half_h, block_w - half_w),
        )
        for sub_row, sub_col, sub_h, sub_w in sub_blocks:
            split_block(sub_row, sub_col, sub_h, sub_w)

    for row in range(0, height, b_max):
        for col in range(0, width, b_max):
            block_h = min(b_max, height - row)
            block_w = min(b_max, width - col)
            split_block(row, col, block_h, block_w)

    return AdaptivePartitionResult(
        source_image=source_image,
        gray_image=gray,
        gradient_method=gradient_method,
        image_shape=source_image.shape,
        gray_shape=gray.shape,
        b_max=b_max,
        b_min=b_min,
        threshold=threshold,
        global_mean_gradient=global_mean_gradient,
        blocks=tuple(blocks),
    )


def extract_block_views(image: np.ndarray, blocks: Iterable[ImageBlock]) -> list[np.ndarray]:
    """Return image views for the provided blocks."""
    image = np.asarray(image)
    return [image[block.row_slice, block.col_slice] for block in blocks]


def build_block_label_map(shape: tuple[int, int], blocks: Iterable[ImageBlock]) -> np.ndarray:
    """Build a label map where each pixel stores the index of its block."""
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")

    label_map = np.full(shape, -1, dtype=np.int32)
    for index, block in enumerate(blocks):
        label_map[block.row_slice, block.col_slice] = index
    return label_map


def _normalize_storage_format(storage_format: str) -> str:
    normalized = storage_format.lower().strip()
    aliases = {
        "uint8_0_3": "uint8_0_3",
        "0_3": "uint8_0_3",
        "uint8_1_4": "uint8_1_4",
        "1_4": "uint8_1_4",
        "char": "char",
        "chars": "char",
        "acgt": "char",
    }
    if normalized not in aliases:
        raise ValueError("storage_format must be one of: 'uint8_0_3', 'uint8_1_4', 'char'")
    return aliases[normalized]


def _dna_to_codes(dna_matrix: np.ndarray, storage_format: str) -> np.ndarray:
    normalized = _normalize_storage_format(storage_format)
    dna_matrix = np.asarray(dna_matrix)
    if normalized == "uint8_0_3":
        return _validate_dna_codes(dna_matrix, name="dna_matrix")
    if normalized == "uint8_1_4":
        dna_codes = np.asarray(dna_matrix, dtype=np.int16) - 1
        return _validate_dna_codes(dna_codes, name="dna_matrix")

    if dna_matrix.dtype.kind not in {"U", "S", "O"}:
        raise ValueError("char DNA matrix must contain A/C/G/T values")
    if not np.all(np.isin(dna_matrix, DNA_CODE_TO_BASE)):
        raise ValueError("char DNA matrix must contain only A/C/G/T values")
    result = np.empty(dna_matrix.shape, dtype=np.uint8)
    for base, code in DNA_BASE_TO_CODE.items():
        result[dna_matrix == base] = code
    return result


def _convert_dna_storage(dna_codes: np.ndarray, storage_format: str) -> np.ndarray:
    normalized = _normalize_storage_format(storage_format)
    dna_codes = _validate_dna_codes(dna_codes, name="dna_codes")
    if normalized == "uint8_0_3":
        return dna_codes
    if normalized == "uint8_1_4":
        return dna_codes + 1
    return DNA_CODE_TO_BASE[dna_codes]


def _validate_dna_codes(
    dna_codes: np.ndarray,
    name: str = "dna_codes",
    require_nine_planes: bool = False,
) -> np.ndarray:
    dna_codes = np.asarray(dna_codes)
    if dna_codes.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer DNA codes in the range [0, 3]")
    if np.any((dna_codes < 0) | (dna_codes > 3)):
        raise ValueError(f"{name} must contain only DNA codes in the range [0, 3]")
    dna_codes = dna_codes.astype(np.uint8, copy=False)
    if require_nine_planes and (dna_codes.ndim != 3 or dna_codes.shape[0] != 9):
        raise ValueError(f"{name} must have shape (9, H, W)")
    return dna_codes


def _dna_codes_to_symbols(dna_codes: np.ndarray) -> np.ndarray:
    dna_codes = _validate_dna_codes(dna_codes, name="dna_codes")
    return DNA_CODE_TO_BASE[dna_codes]


def _dna_symbols_to_codes(dna_symbols: np.ndarray) -> np.ndarray:
    dna_symbols = np.asarray(dna_symbols)
    if dna_symbols.dtype.kind not in {"U", "S", "O"}:
        raise ValueError("DNA symbols must contain A/C/G/T values")
    if not np.all(np.isin(dna_symbols, DNA_CODE_TO_BASE)):
        raise ValueError("DNA symbols must contain only A/C/G/T values")
    result = np.empty(dna_symbols.shape, dtype=np.uint8)
    for base, code in DNA_BASE_TO_CODE.items():
        result[dna_symbols == base] = code
    return result


def _normalize_dna_operation_name(operation_name: str) -> str:
    normalized = operation_name.lower().strip()
    aliases = {
        "add": "add",
        "addition": "add",
        "+": "add",
        "sub": "sub",
        "subtract": "sub",
        "subtraction": "sub",
        "-": "sub",
        "xor": "xor",
        "^": "xor",
    }
    if normalized not in aliases:
        raise ValueError("operation_name must be one of: 'add', 'sub', 'xor'")
    return aliases[normalized]


def _normalize_global_diffusion_scheme(scheme: str) -> str:
    normalized = scheme.lower().strip()
    aliases = {
        "synchronous": "synchronous",
        "together": "synchronous",
        "shared": "synchronous",
        "coupled": "synchronous",
        "independent": "independent",
        "separate": "independent",
        "pixel_independent": "independent",
        "pixel-independent": "independent",
    }
    if normalized not in aliases:
        raise ValueError("scheme must be one of: 'synchronous', 'independent'")
    return aliases[normalized]


def _normalize_global_parallel_mode(parallel_mode: str) -> str:
    normalized = parallel_mode.lower().strip()
    aliases = {
        "sequential_groups": "sequential_groups",
        "grouped": "sequential_groups",
        "sequential": "sequential_groups",
        "whole_batch": "whole_batch",
        "batch": "whole_batch",
        "vectorized": "whole_batch",
    }
    if normalized not in aliases:
        raise ValueError("parallel_mode must be one of: 'sequential_groups', 'whole_batch'")
    return aliases[normalized]


def _normalize_blockwise_diffusion_channel_mode(channel_mode: str) -> str:
    normalized = channel_mode.lower().strip()
    aliases = {
        "together": "together",
        "shared": "together",
        "coupled": "together",
        "broadcast": "together",
        "all": "together",
        "separate": "separate",
        "independent": "separate",
        "per_plane": "separate",
        "planewise": "separate",
        "split": "separate",
    }
    if normalized not in aliases:
        raise ValueError("channel_mode must be one of: 'together', 'separate'")
    return aliases[normalized]


def _apply_dna_operation_codes(left_codes: np.ndarray, right_codes: np.ndarray, operation_name: str) -> np.ndarray:
    normalized_operation = _normalize_dna_operation_name(operation_name)
    left_codes = _validate_dna_codes(left_codes, name="left_codes")
    right_codes = _validate_dna_codes(right_codes, name="right_codes")
    return DNA_OPERATION_CODE_TABLES[normalized_operation][left_codes, right_codes]


def _apply_dna_operation_symbols(left_symbols: np.ndarray, right_symbols: np.ndarray, operation_name: str) -> np.ndarray:
    return _dna_codes_to_symbols(
        _apply_dna_operation_codes(
            _dna_symbols_to_codes(left_symbols),
            _dna_symbols_to_codes(right_symbols),
            operation_name,
        )
    )


def _encode_random_dna_scalar_with_rule(rng: np.random.Generator, rule_id: int) -> np.ndarray:
    random_pairs = rng.integers(0, 2, size=(9, 2), dtype=np.uint8)
    return encode_bitplane_pair_with_rule(
        random_pairs[:, 0],
        random_pairs[:, 1],
        rule_id,
        storage_format="uint8_0_3",
    )


def generate_block_dna_diffusion_operation_ids(
    num_blocks: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one DNA operation id in [1, 3] for each block."""
    if not isinstance(num_blocks, int) or num_blocks < 0:
        raise ValueError("num_blocks must be a non-negative integer")
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.integers(1, 4, size=num_blocks, dtype=np.uint8)


def generate_block_dna_diffusion_column_permutation(num_columns: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one argsort-based column permutation for a flattened DNA block."""
    if num_columns <= 0:
        raise ValueError("num_columns must be positive")
    return np.argsort(rng.random(num_columns), axis=0).astype(np.intp)


def generate_block_dna_diffusion_key_columns(
    block_rule_id: int,
    num_columns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one DNA scalar key column for each flattened block column using the block rule."""
    if num_columns <= 0:
        raise ValueError("num_columns must be positive")
    key_columns = np.empty((9, num_columns), dtype=np.uint8)
    for column_index in range(num_columns):
        key_columns[:, column_index] = _encode_random_dna_scalar_with_rule(rng, block_rule_id)
    return key_columns


def generate_block_dna_diffusion_keys(
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Generate per-block DNA diffusion operation ids, column permutations, and key columns."""
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    operation_ids = generate_block_dna_diffusion_operation_ids(len(blocks), rng=rng)
    column_permutations: list[np.ndarray] = []
    key_columns: list[np.ndarray] = []
    for block_index, block in enumerate(blocks):
        num_columns = block.height * block.width
        column_permutations.append(generate_block_dna_diffusion_column_permutation(num_columns, rng))
        key_columns.append(generate_block_dna_diffusion_key_columns(int(block_rule_ids[block_index]), num_columns, rng))
    return operation_ids, tuple(column_permutations), tuple(key_columns)


def generate_block_dna_diffusion_v2_key_matrix(
    block_height: int,
    block_width: int,
    block_rule_id: int,
    rng: np.random.Generator,
    channel_mode: str = "together",
) -> np.ndarray:
    """Generate one block-sized DNA key matrix using the same rule as the image block."""
    normalized_mode = _normalize_blockwise_diffusion_channel_mode(channel_mode)
    if block_height <= 0 or block_width <= 0:
        raise ValueError("block_height and block_width must be positive")
    if int(block_rule_id) < 1 or int(block_rule_id) > 8:
        raise ValueError("block_rule_id must be in the range [1, 8]")

    if normalized_mode == "together":
        msb_plane = rng.integers(0, 2, size=(block_height, block_width), dtype=np.uint8)
        lsb_plane = rng.integers(0, 2, size=(block_height, block_width), dtype=np.uint8)
        shared_key_plane = encode_bitplane_pair_with_rule(
            msb_plane,
            lsb_plane,
            int(block_rule_id),
            storage_format="uint8_0_3",
        )
        return np.repeat(shared_key_plane[np.newaxis, :, :], 9, axis=0)

    key_matrix = np.empty((9, block_height, block_width), dtype=np.uint8)
    for plane_index in range(9):
        msb_plane = rng.integers(0, 2, size=(block_height, block_width), dtype=np.uint8)
        lsb_plane = rng.integers(0, 2, size=(block_height, block_width), dtype=np.uint8)
        key_matrix[plane_index] = encode_bitplane_pair_with_rule(
            msb_plane,
            lsb_plane,
            int(block_rule_id),
            storage_format="uint8_0_3",
        )
    return key_matrix


def generate_block_rule_ids(
    num_blocks: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one DNA rule id in [1, 8] for each block."""
    if not isinstance(num_blocks, int) or num_blocks < 0:
        raise ValueError("num_blocks must be a non-negative integer")
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")

    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.integers(1, 9, size=num_blocks, dtype=np.uint8)


def generate_block_permutation_indices(
    num_planes: int,
    block_height: int,
    block_width: int,
    rng: np.random.Generator,
) -> BlockPermutationIndices:
    """Generate argsort-based permutation indices for one DNA block."""
    if num_planes <= 0 or block_height <= 0 or block_width <= 0:
        raise ValueError("num_planes, block_height, and block_width must be positive")

    plane_perm = np.argsort(rng.random(num_planes), axis=0).astype(np.intp)
    row_perm = np.argsort(rng.random(block_height), axis=0).astype(np.intp)
    col_perm = np.argsort(rng.random(block_width), axis=0).astype(np.intp)
    return BlockPermutationIndices(
        plane_perm=plane_perm,
        row_perm=row_perm,
        col_perm=col_perm,
    )


def encode_bitplane_pair_with_rule(
    msb_plane: np.ndarray,
    lsb_plane: np.ndarray,
    rule_id: int,
    storage_format: str = "uint8_0_3",
) -> np.ndarray:
    """Encode one 2-bit plane pair into a DNA plane using the selected rule."""
    msb_plane = np.asarray(msb_plane, dtype=np.uint8)
    lsb_plane = np.asarray(lsb_plane, dtype=np.uint8)
    if msb_plane.shape != lsb_plane.shape:
        raise ValueError("msb_plane and lsb_plane must have the same shape")
    if int(rule_id) < 1 or int(rule_id) > 8:
        raise ValueError("rule_id must be in the range [1, 8]")

    pair_values = ((msb_plane & 1) << 1) | (lsb_plane & 1)
    dna_codes = DNA_RULE_TABLE[int(rule_id) - 1][pair_values]
    return _convert_dna_storage(dna_codes, storage_format)


def decode_dna_plane_with_rule(
    dna_plane: np.ndarray,
    rule_id: int,
    storage_format: str = "uint8_0_3",
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one DNA plane into its original bitplane pair using the selected rule."""
    dna_codes = _dna_to_codes(dna_plane, storage_format)
    if int(rule_id) < 1 or int(rule_id) > 8:
        raise ValueError("rule_id must be in the range [1, 8]")

    pair_values = DNA_RULE_TABLE_INV[int(rule_id) - 1][dna_codes]
    msb_plane = (pair_values >> 1).astype(np.uint8)
    lsb_plane = (pair_values & 1).astype(np.uint8)
    return msb_plane, lsb_plane


def blockwise_dna_encode_legacy(
    bitplanes: np.ndarray,
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    pair_indices: tuple[tuple[int, int], ...] = DNA_HIGH6_PAIR_INDICES,
    storage_format: str = "uint8_0_3",
) -> np.ndarray:
    """Encode the high 6 bits of each RGB channel into 9 DNA planes block by block."""
    bitplanes = np.asarray(bitplanes, dtype=np.uint8)
    if bitplanes.ndim != 3 or bitplanes.shape[0] != 24:
        raise ValueError("bitplanes must have shape (24, H, W)")

    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    _, height, width = bitplanes.shape
    dna_codes = np.empty((len(pair_indices), height, width), dtype=np.uint8)

    for block_index, block in enumerate(blocks):
        rule_id = int(block_rule_ids[block_index])
        row_slice = block.row_slice
        col_slice = block.col_slice
        for dna_plane_index, (msb_idx, lsb_idx) in enumerate(pair_indices):
            encoded_region = encode_bitplane_pair_with_rule(
                bitplanes[msb_idx, row_slice, col_slice],
                bitplanes[lsb_idx, row_slice, col_slice],
                rule_id,
                storage_format="uint8_0_3",
            )
            dna_codes[dna_plane_index, row_slice, col_slice] = encoded_region

    return _convert_dna_storage(dna_codes, storage_format)


def blockwise_dna_encode(
    bitplanes: np.ndarray,
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    pair_indices: tuple[tuple[int, int], ...] = DNA_HIGH6_PAIR_INDICES,
    storage_format: str = "uint8_0_3",
) -> np.ndarray:
    """全面向量化优化的 DNA 块编码，消灭双重循环（空间换时间）"""
    bitplanes = np.asarray(bitplanes, dtype=np.uint8)
    if bitplanes.ndim != 3 or bitplanes.shape[0] != 24:
        raise ValueError("bitplanes must have shape (24, H, W)")

    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    _, height, width = bitplanes.shape
    rule_idx_map = np.empty((height, width), dtype=np.uint8)
    for block_index, block in enumerate(blocks):
        rule_idx_map[block.row_slice, block.col_slice] = block_rule_ids[block_index] - 1
    pairs = np.array(pair_indices)
    msb_idx = pairs[:, 0]  # 9 个 MSB 通道索引
    lsb_idx = pairs[:, 1]  # 9 个 LSB 通道索引

    # 瞬间提取出 3D 矩阵空间
    msb_planes = bitplanes[msb_idx] 
    lsb_planes = bitplanes[lsb_idx]  
    pair_values = (msb_planes << 1) | lsb_planes 


    # 核心优化 利用高级索引和广播机制一步到位查表

    # 将 (H, W) 扩展为 (1, H, W)，NumPy 会自动将其广播(Broadcast) 匹配至 (9, H, W)
    rule_idx_broadcasted = rule_idx_map[np.newaxis, :, :]

    # DNA_RULE_TABLE 的 Shape 是 (8, 4)
    # 传入两个相同/可广播形状的坐标张量，NumPy 底层会用 C 语言级循环并发查表
    dna_codes = DNA_RULE_TABLE[rule_idx_broadcasted, pair_values]  # Shape: (9, H, W)

    return _convert_dna_storage(dna_codes, storage_format)
def decode_dna_matrix_to_bitplanes(
    dna_matrix: np.ndarray,
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    original_bitplanes: np.ndarray,
    storage_format: str = "uint8_0_3",
    pair_indices: tuple[tuple[int, int], ...] = DNA_HIGH6_PAIR_INDICES,
) -> np.ndarray:
    """Decode a DNA matrix back into 24 bitplanes and restore the untouched low 2 bits."""
    dna_codes = _dna_to_codes(dna_matrix, storage_format)
    if dna_codes.ndim != 3 or dna_codes.shape[0] != len(pair_indices):
        raise ValueError("dna_matrix must have shape (9, H, W)")

    original_bitplanes = np.asarray(original_bitplanes, dtype=np.uint8)
    if original_bitplanes.shape[0] != 24:
        raise ValueError("original_bitplanes must have shape (24, H, W)")

    restored_bitplanes = original_bitplanes.copy()
    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    for block_index, block in enumerate(blocks):
        row_slice = block.row_slice
        col_slice = block.col_slice
        rule_id = int(block_rule_ids[block_index])
        for dna_plane_index, (msb_idx, lsb_idx) in enumerate(pair_indices):
            msb_plane, lsb_plane = decode_dna_plane_with_rule(
                dna_codes[dna_plane_index, row_slice, col_slice],
                rule_id,
                storage_format="uint8_0_3",
            )
            restored_bitplanes[msb_idx, row_slice, col_slice] = msb_plane
            restored_bitplanes[lsb_idx, row_slice, col_slice] = lsb_plane

    return restored_bitplanes


def permute_dna_block(
    dna_block: np.ndarray,
    permutation_indices: BlockPermutationIndices,
) -> np.ndarray:
    """Permute one DNA block by reading from the original block with 3D indices."""
    dna_block = np.asarray(dna_block)
    if dna_block.ndim != 3:
        raise ValueError("dna_block must have shape (planes, rows, cols)")
    return dna_block[np.ix_(permutation_indices.plane_perm, permutation_indices.row_perm, permutation_indices.col_perm)]


def permute_dna_blocks(
    dna_matrix: np.ndarray,
    blocks: Iterable[ImageBlock],
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[BlockPermutationIndices, ...]]:
    """Apply per-block 3D permutations to a DNA matrix."""
    dna_matrix = np.asarray(dna_matrix)
    if dna_matrix.ndim != 3:
        raise ValueError("dna_matrix must have shape (planes, H, W)")

    encrypted_matrix = dna_matrix.copy()
    permutation_indices: list[BlockPermutationIndices] = []
    num_planes = dna_matrix.shape[0]

    for block in tuple(blocks):
        indices = generate_block_permutation_indices(num_planes, block.height, block.width, rng)
        source_block = dna_matrix[:, block.row_slice, block.col_slice]
        encrypted_block = permute_dna_block(source_block, indices)
        encrypted_matrix[:, block.row_slice, block.col_slice] = encrypted_block
        permutation_indices.append(indices)

    return encrypted_matrix, tuple(permutation_indices)

def permute_dna_blocks_v2(
    dna_matrix: np.ndarray,
    blocks: Iterable[ImageBlock],
    rd_matrix: np.ndarray,
) -> np.ndarray:
    """
    使用2D一维化混沌随机矩阵 RdMatrix (29, 512*512) 的前 9 个通道
    对每个自适应块进行全维度 3D 像素级加密置乱
    """
    dna_matrix = np.asarray(dna_matrix)
    rd_matrix = np.asarray(rd_matrix)
    
    _, img_height, img_width = dna_matrix.shape
    encrypted_matrix = dna_matrix.copy()
    
    for block in blocks:
        h, w = block.height, block.width
        
        # 1. 提取当前块对应的 DNA 碱基数据 (9, h, w)
        sub_block = dna_matrix[:, block.row_slice, block.col_slice]
        
        # 2. 【核心修复】计算当前块所有像素在 512*512 展平后的一维空间索引
        # r_coords 形状为 (h, 1), c_coords 形状为 (w,) -> 广播后形状为 (h, w)
        r_coords = np.arange(block.row, block.row + h)[:, np.newaxis]
        c_coords = np.arange(block.col, block.col + w)
        flat_spatial_indices = (r_coords * img_width + c_coords).ravel()
        
        # 3. 从 2D 的 rd_matrix 中切出前 9 个通道对应的随机数，形状为 (9, h * w)
        sub_rd = rd_matrix[0:9, flat_spatial_indices]
        
        # 4. 三维联合展平 (由于 sub_block 和 sub_rd 都是行优先展开，它们像素点是一一对应的)
        flat_block = sub_block.ravel()
        flat_rd = sub_rd.ravel()
        
        # 5. 生成全维度一维置乱索引 (长度为 9 * h * w)
        perm_indices = np.argsort(flat_rd)
        
        # 6. 执行位置重排并回填
        scrambled_flat = flat_block[perm_indices]
        encrypted_matrix[:, block.row_slice, block.col_slice] = scrambled_flat.reshape(9, h, w)
        
    return encrypted_matrix



def apply_blockwise_dna_diffusion(
    dna_matrix: np.ndarray,
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    storage_format: str = "uint8_0_3",
) -> BlockDNADiffusionResult:
    """Apply intra-block DNA diffusion with per-block operation selection and scalar DNA keys."""
    normalized_format = _normalize_storage_format(storage_format)
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    dna_codes = _dna_to_codes(dna_matrix, normalized_format)
    dna_codes = _validate_dna_codes(dna_codes, name="dna_matrix", require_nine_planes=True)

    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    operation_ids, column_permutations, key_columns = generate_block_dna_diffusion_keys(
        blocks,
        block_rule_ids,
        rng=rng,
    )
    diffused_codes = dna_codes.copy()

    for block_index, block in enumerate(blocks):
        row_slice = block.row_slice
        col_slice = block.col_slice
        source_block = dna_codes[:, row_slice, col_slice]
        _, block_height, block_width = source_block.shape
        flat_block_codes = source_block.reshape(9, block_height * block_width)
        permutation = column_permutations[block_index]
        key_block = key_columns[block_index]
        operation_name = DNA_OPERATION_ID_TO_NAME[int(operation_ids[block_index])]
        num_columns = permutation.size
        diffused_block_codes = np.empty_like(flat_block_codes)

        for column_index in range(num_columns):
            prev_column = flat_block_codes[:, permutation[column_index - 1]]
            curr_column = flat_block_codes[:, permutation[column_index]]
            key_column_codes = key_block[:, column_index]
            mixed_column = _apply_dna_operation_codes(prev_column, curr_column, operation_name)
            diffused_block_codes[:, column_index] = _apply_dna_operation_codes(
                mixed_column,
                key_column_codes,
                operation_name,
            )

        diffused_codes[:, row_slice, col_slice] = diffused_block_codes.reshape(9, block_height, block_width)

    return BlockDNADiffusionResult(
        dna_matrix=_convert_dna_storage(diffused_codes, normalized_format),
        block_operation_ids=operation_ids,
        block_column_permutations=column_permutations,
        block_key_columns=key_columns,
        storage_format=normalized_format,
    )


def apply_blockwise_dna_diffusionV2(
    dna_matrix: np.ndarray,
    blocks: Iterable[ImageBlock],
    block_rule_ids: np.ndarray,
    operation_name: str = "xor",
    channel_mode: str = "together",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    storage_format: str = "uint8_0_3",
) -> BlockDNADiffusionV2Result:
    """Apply a simpler intra-block DNA diffusion of the form DNA'_i = DNA_i & r_i."""
    normalized_format = _normalize_storage_format(storage_format)
    normalized_operation = _normalize_dna_operation_name(operation_name)
    normalized_channel_mode = _normalize_blockwise_diffusion_channel_mode(channel_mode)
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    dna_codes = _dna_to_codes(dna_matrix, normalized_format)
    dna_codes = _validate_dna_codes(dna_codes, name="dna_matrix", require_nine_planes=True)

    blocks = tuple(blocks)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
    if block_rule_ids.shape != (len(blocks),):
        raise ValueError("block_rule_ids shape must be (len(blocks),)")

    diffused_codes = dna_codes.copy()
    key_matrices: list[np.ndarray] = []

    for block_index, block in enumerate(blocks):
        row_slice = block.row_slice
        col_slice = block.col_slice
        source_block_codes = dna_codes[:, row_slice, col_slice]
        key_matrix = generate_block_dna_diffusion_v2_key_matrix(
            block.height,
            block.width,
            int(block_rule_ids[block_index]),
            rng,
            channel_mode=normalized_channel_mode,
        )
        diffused_codes[:, row_slice, col_slice] = _apply_dna_operation_codes(
            source_block_codes,
            key_matrix,
            normalized_operation,
        )
        key_matrices.append(key_matrix)

    return BlockDNADiffusionV2Result(
        dna_matrix=_convert_dna_storage(diffused_codes, normalized_format),
        block_key_matrices=tuple(key_matrices),
        channel_mode=normalized_channel_mode,
        operation_name=normalized_operation,
        storage_format=normalized_format,
    )


def generate_global_dna_diffusion_permutation(num_pixels: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one argsort-based global diffusion permutation over H*W spatial positions."""
    if num_pixels <= 0:
        raise ValueError("num_pixels must be positive")
    return np.argsort(rng.random(num_pixels), axis=0).astype(np.intp)


def generate_global_dna_diffusion_key_matrix(
    height: int,
    width: int,
    rng: np.random.Generator,
    scheme: str = "synchronous",
) -> tuple[np.ndarray, int]:
    """Generate a global DNA key matrix using a single global random DNA rule."""
    normalized_scheme = _normalize_global_diffusion_scheme(scheme)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    key_rule_id = int(rng.integers(1, 9, dtype=np.uint8))
    if normalized_scheme == "synchronous":
        msb_plane = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
        lsb_plane = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
        shared_key_plane = encode_bitplane_pair_with_rule(
            msb_plane,
            lsb_plane,
            key_rule_id,
            storage_format="uint8_0_3",
        )
        return np.repeat(shared_key_plane[np.newaxis, :, :], 9, axis=0), key_rule_id

    key_matrix = np.empty((9, height, width), dtype=np.uint8)
    for plane_index in range(9):
        msb_plane = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
        lsb_plane = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
        key_matrix[plane_index] = encode_bitplane_pair_with_rule(
            msb_plane,
            lsb_plane,
            key_rule_id,
            storage_format="uint8_0_3",
        )
    return key_matrix, key_rule_id


def apply_global_dna_diffusion(
    dna_matrix: np.ndarray,
    operation_name: str = "xor",
    scheme: str = "synchronous",
    parallel_size: int = 512,
    parallel_mode: str = "whole_batch",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    storage_format: str = "uint8_0_3",
) -> GlobalDNADiffusionResult:
    """Apply global DNA diffusion on the post-block-diffusion DNA image.
        scheme="synchronous"  时 9 通道共享同一个 (H,W)
              ="independent"  时 9 通道各自有独立 key plane,但仍使用同一个全局 rule
        parallel_mode="sequential_groups" 时 按照 permutation_indices 定义的顺序分组串行处理每个像素位置
                     ="whole_batch"     时 将所有像素位置视为一个批次整体进行向量化处理（需要更多内存）
    """
    normalized_format = _normalize_storage_format(storage_format)
    normalized_operation = _normalize_dna_operation_name(operation_name)
    if normalized_operation != "xor":
        raise ValueError("apply_global_dna_diffusion currently supports only xor, matching the requested formula")

    normalized_scheme = _normalize_global_diffusion_scheme(scheme)
    normalized_parallel_mode = _normalize_global_parallel_mode(parallel_mode)
    if not isinstance(parallel_size, int) or parallel_size <= 0:
        raise ValueError("parallel_size must be a positive integer")
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    dna_codes = _dna_to_codes(dna_matrix, normalized_format)
    dna_codes = _validate_dna_codes(dna_codes, name="dna_matrix", require_nine_planes=True)

    _, height, width = dna_codes.shape
    num_pixels = height * width
    permutation_indices = generate_global_dna_diffusion_permutation(num_pixels, rng)
    key_matrix, key_rule_id = generate_global_dna_diffusion_key_matrix(
        height,
        width,
        rng,
        scheme=normalized_scheme,
    )
    img_flat = dna_codes.reshape(9, num_pixels)
    key_flat = key_matrix.reshape(9, num_pixels)
    current_columns = img_flat[:, permutation_indices]
    key_columns = key_flat[:, permutation_indices]

    if normalized_parallel_mode == "whole_batch":
        previous_columns = img_flat[:, np.roll(permutation_indices, 1)]
        mixed_columns = _apply_dna_operation_codes(current_columns, previous_columns, normalized_operation)
        diffused_flat = _apply_dna_operation_codes(mixed_columns, key_columns, normalized_operation)

    else:
        diffused_flat = np.empty_like(current_columns)
        carry_column = img_flat[:, permutation_indices[-1]][:, np.newaxis]
        for start in range(0, num_pixels, parallel_size):
            end = min(start + parallel_size, num_pixels)
            group_current = current_columns[:, start:end]
            group_keys = key_columns[:, start:end]
            group_carry = np.repeat(carry_column, end - start, axis=1)
            mixed_group = _apply_dna_operation_codes(group_current, group_carry, normalized_operation)
            diffused_flat[:, start:end] = _apply_dna_operation_codes(mixed_group, group_keys, normalized_operation)
            carry_column = img_flat[:, permutation_indices[end - 1]][:, np.newaxis]

    diffused_codes = diffused_flat.reshape(9, height, width)
    return GlobalDNADiffusionResult(
        dna_matrix=_convert_dna_storage(diffused_codes, normalized_format),
        permutation_indices=permutation_indices,
        key_matrix=key_matrix,
        key_rule_id=key_rule_id,
        scheme=normalized_scheme,
        parallel_mode=normalized_parallel_mode,
        parallel_size=parallel_size,
        operation_name=normalized_operation,
        storage_format=normalized_format,
    )


def shuffle_blocks_between_groups(
    dna_matrix: np.ndarray,
    bitplanes: np.ndarray,
    block_rule_ids: np.ndarray,
    blocks: Iterable[ImageBlock],
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    storage_format: str = "uint8_0_3",
) -> BlockShuffleResult:
    """Shuffle same-sized blocks as atomic units and keep outputs decodable."""
    normalized_format = _normalize_storage_format(storage_format)
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    dna_codes = _dna_to_codes(dna_matrix, normalized_format)
    bitplanes = np.asarray(bitplanes, dtype=np.uint8)
    block_rule_ids = np.asarray(block_rule_ids, dtype=np.uint8).copy()
    blocks = tuple(blocks)

    shuffled_dna = dna_codes.copy()
    shuffled_bitplanes = bitplanes.copy()
    shuffled_rule_ids = block_rule_ids.copy()
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for index, block in enumerate(blocks):
        grouped_indices.setdefault((block.height, block.width), []).append(index)

    groups: list[BlockShuffleGroup] = []
    for block_shape, target_indices_list in grouped_indices.items():
        target_indices = np.asarray(target_indices_list, dtype=np.intp)
        if target_indices.size <= 1:
            groups.append(
                BlockShuffleGroup(
                    block_shape=block_shape,
                    target_block_indices=target_indices,
                    source_block_indices=target_indices.copy(),
                )
            )
            continue

        source_block_indices = target_indices[np.argsort(rng.random(target_indices.size), axis=0)]
        for target_index, source_index in zip(target_indices, source_block_indices, strict=True):
            target_block = blocks[int(target_index)]
            source_block = blocks[int(source_index)]
            shuffled_dna[:, target_block.row_slice, target_block.col_slice] = dna_codes[:, source_block.row_slice, source_block.col_slice]
            shuffled_rule_ids[int(target_index)] = block_rule_ids[int(source_index)]

            for plane_index in ENCODED_HIGH6_BITPLANE_INDICES + LOW2_BITPLANE_INDICES:
                shuffled_bitplanes[plane_index, target_block.row_slice, target_block.col_slice] = bitplanes[plane_index, source_block.row_slice, source_block.col_slice]

        groups.append(
            BlockShuffleGroup(
                block_shape=block_shape,
                target_block_indices=target_indices,
                source_block_indices=source_block_indices,
            )
        )

    return BlockShuffleResult(
        dna_matrix=_convert_dna_storage(shuffled_dna, normalized_format),
        block_rule_ids=shuffled_rule_ids,
        bitplanes=shuffled_bitplanes,
        groups=tuple(groups),
        storage_format=normalized_format,
    )


def encode_image_blocks_to_dna(
    image: np.ndarray,
    blocks: Iterable[ImageBlock],
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    storage_format: str = "uint8_0_3",
    pair_indices: tuple[tuple[int, int], ...] = DNA_HIGH6_PAIR_INDICES,
    rd_matrix: np.ndarray | None = None,
    verbose: bool = True,
) -> DNAEncodingResult:
    """Encode an RGB image into a block-wise DNA matrix of shape (9, H, W)."""
    normalized_format = _normalize_storage_format(storage_format)
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    blocks = tuple(blocks)
    st = time.time()
    bitplanes = image_to_bitplanes(image)
    if verbose:
        print(f"Bitplane extraction took {time.time() - st:.6f} seconds")

    blocks_num = len(blocks)
    if rd_matrix is None:
        rd_matrix = globals().get("RdMatrix")
    if rd_matrix is None:
        block_rule_ids = generate_block_rule_ids(blocks_num, rng=rng)
    else:
        rd_matrix = np.asarray(rd_matrix)
        if rd_matrix.ndim != 2 or rd_matrix.shape[1] < blocks_num:
            raise ValueError("rd_matrix must have shape (N, L) with L >= len(blocks)")
        block_rule_ids = (np.mod(np.floor(rd_matrix[-1, :blocks_num] * 1e10), 8) + 1).astype(np.uint8)

    st = time.time()
    encoded_dna_codes = blockwise_dna_encode(
        bitplanes,
        blocks,
        block_rule_ids,
        pair_indices=pair_indices,
        storage_format="uint8_0_3",
    )
    if verbose:
        print(f"Encoding took {time.time() - st:.6f} seconds")

    return DNAEncodingResult(
        bitplanes=bitplanes,
        encoded_dna_matrix=_convert_dna_storage(encoded_dna_codes, normalized_format),
        dna_matrix = None,
        block_rule_ids=block_rule_ids,
        pair_indices=pair_indices,
        blocks=blocks,
        storage_format=normalized_format,
    )




def encrypt_image_array(image: str | Path | np.ndarray | pil_image.Image, **kwargs) -> np.ndarray:
    """Backward-friendly alias for encrypt_image."""
    return encrypt_image(image, **kwargs)


def show_images(original: np.ndarray, reconstructed: np.ndarray, title_suffix: str = "") -> None:
    """Display the original and reconstructed images side by side."""
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed)
    axes[1].set_title(f"Reconstructed{title_suffix}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_image_comparison(
    original: np.ndarray,
    no_block_shuffle_image: np.ndarray,
    with_block_shuffle_image: np.ndarray,
    with_block_shuffle_and_diffusion_image: np.ndarray,
    with_global_diffusion_image: np.ndarray | None = None,
    diffusion_title: str = "Encrypted with Block Shuffle + Diffusion",
    global_diffusion_title: str = "Encrypted with Global Diffusion",
) -> None:
    """Display original, shuffled, block-diffused, and optionally global-diffused images together."""
    images = [
        original,
        no_block_shuffle_image,
        with_block_shuffle_image,
        with_block_shuffle_and_diffusion_image,
    ]
    titles = [
        "Original",
        "Encrypted without Block Shuffle",
        "Encrypted with Block Shuffle",
        diffusion_title,
    ]
    if with_global_diffusion_image is not None:
        images.append(with_global_diffusion_image)
        titles.append(global_diffusion_title)

    _, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
    axes = np.atleast_1d(axes)
    for ax, image, title in zip(axes, images, titles, strict=True):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_partition_overlay(
    image: np.ndarray,
    blocks: Iterable[ImageBlock],
    *,
    ax: plt.Axes | None = None,
    line_color: str = "red",
    line_width: float = 0.6,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Display the adaptive partition result as red block boundaries over the image."""
    image = np.asarray(image)
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    cmap = "gray" if image.ndim == 2 else None
    ax.imshow(image, cmap=cmap)

    for block in blocks:
        rect = Rectangle(
            (block.col - 0.5, block.row - 0.5),
            block.width,
            block.height,
            fill=False,
            edgecolor=line_color,
            linewidth=line_width,
        )
        ax.add_patch(rect)

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    if show and created_ax:
        plt.show()
    return fig, ax


def save_partition_overlay(
    image: np.ndarray,
    blocks: Iterable[ImageBlock],
    save_path: str | Path,
    *,
    dpi: int = 300,
    line_color: str = "red",
    line_width: float = 0.6,
) -> Path:
    """Save the adaptive partition overlay image to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_partition_overlay(
        image,
        blocks,
        line_color=line_color,
        line_width=line_width,
        show=False,
    )
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path


def print_encryption_profile(stage_timings: list[tuple[str, float]]) -> None:
    """Print encryption stage timings and percentage breakdown."""
    total_time = sum(seconds for _, seconds in stage_timings)
    if total_time <= 0:
        print("[profile] no timing data available")
        return

    print("\nEncryption performance profile")
    print("-" * 72)
    print(f"{'Stage':<42} {'Time(s)':>12} {'Percent':>10}")
    print("-" * 72)
    for stage_name, seconds in stage_timings:
        percent = seconds / total_time * 100.0
        print(f"{stage_name:<42} {seconds:>12.6f} {percent:>9.2f}%")
    print("-" * 72)
    print(f"{'Total':<42} {total_time:>12.6f} {100.0:>9.2f}%\n")


def plot_encryption_profile(
    stage_timings: list[tuple[str, float]],
    *,
    title: str = "Encryption Stage Time Cost",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize encryption stage timings as a horizontal percentage bar chart."""
    total_time = sum(seconds for _, seconds in stage_timings)
    if total_time <= 0:
        raise ValueError("stage_timings must contain positive timing values")

    stage_names = [stage_name for stage_name, _ in stage_timings]
    seconds = np.array([seconds for _, seconds in stage_timings], dtype=np.float64)
    percentages = seconds / total_time * 100.0

    fig_height = max(4.0, 0.45 * len(stage_names) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y_pos = np.arange(len(stage_names))
    bars = ax.barh(y_pos, percentages, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(stage_names))))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stage_names)
    ax.invert_yaxis()
    ax.set_xlabel("Time cost (%)")
    ax.set_title(f"{title}  |  total = {total_time:.4f}s")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    label_offset = max(float(percentages.max()) * 0.01, 0.2) if percentages.size else 0.2
    for bar, sec, pct in zip(bars, seconds, percentages, strict=True):
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}% ({sec:.4f}s)",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(float(percentages.max()) * 1.25, 5.0))
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _load_rgb_image_array(image: str | Path | np.ndarray | pil_image.Image) -> np.ndarray:
    """Load a path/PIL image/ndarray into an RGB uint8 ndarray."""
    if isinstance(image, (str, Path)):
        with pil_image.open(image) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    if isinstance(image, pil_image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return np.asarray(pil_image.fromarray(image_array).convert("RGB"), dtype=np.uint8)
    if image_array.ndim != 3:
        raise ValueError("image must be a path, PIL image, 2D grayscale array, or 3D image array")

    if image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)
    elif image_array.shape[2] >= 3:
        image_array = image_array[..., :3]
    else:
        raise ValueError("3D image array must have at least one channel")

    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating) and image_array.size and np.nanmax(image_array) <= 1.0:
            image_array = np.clip(image_array, 0.0, 1.0) * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_array)


def encrypt_image(
    image: str | Path | np.ndarray | pil_image.Image,
    *,
    params: dict[str, float] | None = None,
    seed: int = 2026,
    b_max: int = 64,
    b_min: int = 12,
    gradient_method: str = "gray",
    gradient_threshold: float | None = None,
    storage_format: str = "uint8_0_3",
    diffusion_operation_name: str = "xor",
    diffusion_channel_mode: str = "together",
    global_diffusion_scheme: str = "independent",
    global_parallel_mode: str = "sequential_groups",
    global_parallel_size: int = 512,
    verbose: bool = False,
    visualize_profile: bool = False,
) -> np.ndarray:
    """Apply Shuffled and Diffused encryption to an image using the same pipeline as the module demo main.

    Parameters
    ----------
    image:
        Input image path, PIL image, or numpy array. Arrays are converted to RGB uint8.
    visualize_profile:
        Whether to show a matplotlib bar chart of each encryption stage's time percentage.
        A textual timing table is always printed.

    Returns
    -------
    np.ndarray
        Shuffled and Diffused RGB image with dtype uint8 and shape (H, W, 3).
    """
    stage_timings: list[tuple[str, float]] = []

    def finish_stage(stage_name: str, start_time: float) -> None:
        stage_timings.append((stage_name, time.perf_counter() - start_time))

    st = time.perf_counter()
    img_arr = _load_rgb_image_array(image)
    finish_stage("Image loading / RGB conversion", st)

    height, width = img_arr.shape[:2]
    L = height * width


    if params is None:
        params = {
            "mu": 5,
            "lam": 5,
            "a": 100,
            "b": 200,
            "xi": 1,
            "eta": 1,
        }

    st = time.perf_counter()
    seed_rng = np.random.default_rng(seed)
    x0 = seed_rng.random(L)
    z0 = float(seed_rng.random())
    cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0, "z0": z0})
    rd_matrix = cml.generate_rdseq_fast(28)
    finish_stage("CML random matrix generation", st)
    print(f"CML random matrix generation took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    partition_result = adaptive_partition(
        img_arr,
        b_max=b_max,
        b_min=b_min,
        gradient_threshold=gradient_threshold,
        gradient_method=gradient_method,
    )
    finish_stage("Adaptive partition", st)
    if verbose:
        print(f"partitioning took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    dna_result = encode_image_blocks_to_dna(
        partition_result.source_image,
        partition_result.blocks,
        seed=seed,
        storage_format=storage_format,
        rd_matrix=rd_matrix,
        verbose=verbose,
    )
    finish_stage("DNA encoding", st)
    if verbose:
        print(f"DNA encoding took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    dna_result.dna_matrix = permute_dna_blocks_v2(dna_result.encoded_dna_matrix, dna_result.blocks, rd_matrix)
    finish_stage("DNA intra-block permutation", st)
    if verbose:
        print(f"DNA Block inter-block permutation took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    block_shuffle_result = shuffle_blocks_between_groups(
        dna_result.dna_matrix,
        dna_result.bitplanes,
        dna_result.block_rule_ids,
        dna_result.blocks,
        seed=seed + 1,
        storage_format=dna_result.storage_format,
    )
    finish_stage("Same-size block shuffle", st)
    if verbose:
        print(f"grouping shuffled took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    diffusion_result = apply_blockwise_dna_diffusionV2(
        block_shuffle_result.dna_matrix,
        dna_result.blocks,
        block_shuffle_result.block_rule_ids,
        operation_name=diffusion_operation_name,
        channel_mode=diffusion_channel_mode,
        seed=seed + 2,
        storage_format=block_shuffle_result.storage_format,
    )
    finish_stage("Blockwise DNA diffusion V2", st)
    if verbose:
        print(f"blockwise DNA diffusion V2 took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    global_diffusion_result = apply_global_dna_diffusion(
        diffusion_result.dna_matrix,
        operation_name="xor",
        scheme=global_diffusion_scheme,
        parallel_size=global_parallel_size,
        parallel_mode=global_parallel_mode,
        seed=seed + 3,
        storage_format=diffusion_result.storage_format,
    )
    finish_stage("Global DNA diffusion", st)
    if verbose:
        print(f"global DNA diffusion took {stage_timings[-1][1]:.6f} seconds")

    st = time.perf_counter()
    encrypted_bitplanes = decode_dna_matrix_to_bitplanes(
        global_diffusion_result.dna_matrix,
        dna_result.blocks,
        block_shuffle_result.block_rule_ids,
        block_shuffle_result.bitplanes,
        storage_format=global_diffusion_result.storage_format,
        pair_indices=dna_result.pair_indices,
    )
    finish_stage("DNA decoding to bitplanes", st)

    st = time.perf_counter()
    encrypted_image = bitplanes_to_image(encrypted_bitplanes)
    finish_stage("Bitplanes to RGB image", st)

    print_encryption_profile(stage_timings)
    if visualize_profile:
        plot_encryption_profile(stage_timings)

    return encrypted_image


# Backward-compatible alias for older callers.
SD_image = encrypt_image


if __name__ == "__main__":
    img_pth = Path(r"C:\ImageEncryptionV2\image\img3.png")
    encrypted_img = encrypt_image(img_pth,verbose=True)
    show_images(_load_rgb_image_array(img_pth), encrypted_img, title_suffix=" (Encrypted)")
    # img = pil_image.open(img_pth).convert("RGB")
    # img_arr = np.asarray(img, dtype=np.uint8)
    # H,W = img_arr.shape[:2]

    # L = H * W
    # params = {
    #     "mu": 5,
    #     "lam": 5,
    #     "a": 100,
    #     "b": 200,
    #     "xi": 1,
    #     "eta": 1,
    # }
    # seed = 2026
    # np.random.seed(seed)
    # x0 = np.random.rand(L)
    # z0 = np.random.rand()
    # cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0, "z0": z0})
    # # global RdMatrix
    # RdMatrix = cml.generate_rdseq(28)

    # # 自适应分块
    # st = time.time()
    # partition_result = adaptive_partition(img_arr, b_max=64, b_min=12, gradient_method="gray")
    # print(f"partitioning took {time.time() - st:.6f} seconds")

    # # DNA 编码
    # st = time.time()
    # dna_result = encode_image_blocks_to_dna(
    #     partition_result.source_image,
    #     partition_result.blocks,
    #     seed=2026,
    #     storage_format="uint8_0_3",
    # )
    # print(f"DNA encoding took {time.time() - st:.6f} seconds")
    
    # st = time.time()
    # dna_result.dna_matrix = permute_dna_blocks_v2(dna_result.encoded_dna_matrix, dna_result.blocks, RdMatrix)
    # print(f"DNA Block inter-block permutation took {time.time() - st:.6f} seconds")


    # st = time.time()
    # reconstructed_bitplanes_no_shuffle = decode_dna_matrix_to_bitplanes(
    #     dna_result.dna_matrix,
    #     dna_result.blocks,
    #     dna_result.block_rule_ids,
    #     dna_result.bitplanes,
    #     storage_format=dna_result.storage_format,
    #     pair_indices=dna_result.pair_indices,
    # )
    # reconstructed_image_no_shuffle = bitplanes_to_image(reconstructed_bitplanes_no_shuffle)

    # st = time.time()
    # block_shuffle_result = shuffle_blocks_between_groups(
    #     dna_result.dna_matrix,
    #     dna_result.bitplanes,
    #     dna_result.block_rule_ids,
    #     dna_result.blocks,
    #     seed=2027,
    #     storage_format=dna_result.storage_format,
    # )
    # print(f"grouping shuffled took {time.time() - st:.6f} seconds")

    # st = time.time()
    # reconstructed_bitplanes = decode_dna_matrix_to_bitplanes(
    #     block_shuffle_result.dna_matrix,
    #     dna_result.blocks,
    #     block_shuffle_result.block_rule_ids,
    #     block_shuffle_result.bitplanes,
    #     storage_format=block_shuffle_result.storage_format,
    #     pair_indices=dna_result.pair_indices,
    # )
    # reconstructed_image = bitplanes_to_image(reconstructed_bitplanes)

    # diffusion_operation_name = "xor"
    # diffusion_channel_mode = "together"

    # st = time.time()
    # diffusion_result = apply_blockwise_dna_diffusionV2(
    #     block_shuffle_result.dna_matrix,
    #     dna_result.blocks,
    #     block_shuffle_result.block_rule_ids,
    #     operation_name=diffusion_operation_name,
    #     channel_mode=diffusion_channel_mode,
    #     seed=2028,
    #     storage_format=block_shuffle_result.storage_format,
    # )
    # print(f"blockwise DNA diffusion V2 took {time.time() - st:.6f} seconds")

    # st = time.time()
    # reconstructed_bitplanes_with_diffusion = decode_dna_matrix_to_bitplanes(
    #     diffusion_result.dna_matrix,
    #     dna_result.blocks,
    #     block_shuffle_result.block_rule_ids,
    #     block_shuffle_result.bitplanes,
    #     storage_format=diffusion_result.storage_format,
    #     pair_indices=dna_result.pair_indices,
    # )
    # reconstructed_image_with_diffusion = bitplanes_to_image(reconstructed_bitplanes_with_diffusion)
    # print(f"DNA decoding with inter-block shuffle and diffusion V2 took {time.time() - st:.6f} seconds")

    # global_diffusion_scheme = "independent"
    # global_parallel_mode = "sequential_groups"
    # global_parallel_size = 64

    # st = time.time()
    # global_diffusion_result = apply_global_dna_diffusion(
    #     diffusion_result.dna_matrix,
    #     operation_name="xor",
    #     scheme=global_diffusion_scheme,
    #     parallel_size=global_parallel_size,
    #     parallel_mode=global_parallel_mode,
    #     seed=2029,
    #     storage_format=diffusion_result.storage_format,
    # )
    # print(f"global DNA diffusion took {time.time() - st:.6f} seconds")

    # st = time.time()
    # reconstructed_bitplanes_with_global_diffusion = decode_dna_matrix_to_bitplanes(
    #     global_diffusion_result.dna_matrix,
    #     dna_result.blocks,
    #     block_shuffle_result.block_rule_ids,
    #     block_shuffle_result.bitplanes,
    #     storage_format=global_diffusion_result.storage_format,
    #     pair_indices=dna_result.pair_indices,
    # )
    # reconstructed_image_with_global_diffusion = bitplanes_to_image(reconstructed_bitplanes_with_global_diffusion)
    # print(f"DNA decoding with inter-block shuffle, diffusion V2, and global diffusion took {time.time() - st:.6f} seconds")

    # output_path = Path("output") / f"adaptive_partition_overlay_{partition_result.gradient_method}.png"
    # save_partition_overlay(partition_result.source_image, partition_result.blocks, output_path, dpi=300, line_width=0.7)

    # rule_hist = np.bincount(block_shuffle_result.block_rule_ids, minlength=9)[1:]

    # first_diffusion_key_matrix = diffusion_result.block_key_matrices[0] if diffusion_result.block_key_matrices else None


    # show_image_comparison(
    #     partition_result.source_image,
    #     reconstructed_image_no_shuffle,
    #     reconstructed_image,
    #     reconstructed_image_with_diffusion,
    #     with_global_diffusion_image=reconstructed_image_with_global_diffusion,
    #     diffusion_title=f"Encrypted with Block Shuffle + Diffusion V2 ({diffusion_result.channel_mode}, {diffusion_result.operation_name})",
    #     global_diffusion_title=(
    #         f"Encrypted with Global Diffusion "
    #         f"({global_diffusion_result.scheme}, {global_diffusion_result.parallel_mode}, "
    #         f"n={global_diffusion_result.parallel_size}, {global_diffusion_result.operation_name})"
    #     ),
    # )
