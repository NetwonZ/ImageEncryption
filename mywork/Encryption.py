"""Reversible adaptive DNA image encryption.

The module deliberately keeps all encryption random material in
``EncryptionMetadata``.  The current source is NumPy's pseudorandom generator;
the marked locations are the single replacement points for a future chaotic
key-stream generator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pil_image
from .SalomonCouplingCML import SalomoncouplingCML

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
DNA_PAIR_INDICES = tuple((index, index + 1) for index in range(0, 24, 2))
DNA_PLANE_COUNT = len(DNA_PAIR_INDICES)
DNA_12_BIT_LUT = (
    (
        np.arange(4096, dtype=np.uint16)[:, np.newaxis]
        >> np.arange(11, -1, -1, dtype=np.uint16)[np.newaxis, :]
    )
    & 1
).astype(np.uint8)
DNA_OPERATION_RULE_ID_TO_NAME = {
    1: "add",
    2: "sub",
    3: "xor",
}


@dataclass(frozen=True)
class ImageBlock:
    """One rectangular block in the adaptive partition."""

    row: int
    col: int
    height: int
    width: int
    mean_gradient: float

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.width)


@dataclass(frozen=True)
class EncryptionConfig:
    """Public parameters of the reversible encryption pipeline."""

    seed: int = 2026
    b_max: int = 64
    b_min: int = 12
    gradient_method: str = "gray"
    gradient_threshold: float | None = None
    block_operation: str = "xor"
    global_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.b_max <= 0 or self.b_min <= 0 or self.b_min > self.b_max:
            raise ValueError("b_max and b_min must be positive and b_min must not exceed b_max")
        if self.block_operation.lower().strip() not in {"xor", "add", "sub"}:
            raise ValueError("block_operation must be one of: 'xor', 'add', 'sub'")
        if not isinstance(self.global_parallel_size, int) or self.global_parallel_size <= 0:
            raise ValueError("global_parallel_size must be a positive integer")


@dataclass(frozen=True)
class BlockShuffleKey:
    """Forward mapping: target block receives the data from source block."""

    block_shape: tuple[int, int]
    target_block_indices: np.ndarray
    source_block_indices: np.ndarray


@dataclass(frozen=True)
class BlockAxisPermutationIndices:
    """Independent permutation indices for the DNA-plane, row, and column axes."""

    plane_permutation: np.ndarray
    row_permutation: np.ndarray
    col_permutation: np.ndarray


@dataclass(frozen=True)
class EncryptionKeyMaterial:
    """All key and state material required by both encryption and decryption."""

    original_block_rule_ids: np.ndarray
    shuffled_block_rule_ids: np.ndarray
    intra_block_permutations: tuple[BlockAxisPermutationIndices, ...]
    block_shuffle_keys: tuple[BlockShuffleKey, ...]
    block_key_matrices: tuple[np.ndarray, ...]
    block_operation_rule_ids: np.ndarray
    global_permutation: np.ndarray
    global_key_matrix: np.ndarray
    global_initial_vector: np.ndarray
    global_key_rule_id: np.ndarray


@dataclass(frozen=True)
class EncryptionMetadata:
    """Metadata that must accompany the ciphertext for lossless decryption."""

    version: str
    image_shape: tuple[int, int, int]
    blocks: tuple[ImageBlock, ...]
    config: EncryptionConfig
    key_material: EncryptionKeyMaterial


@dataclass(frozen=True)
class EncryptionProfile:
    """Per-stage timing information displayed in the requested tabular form."""

    title: str
    stage_timings: tuple[tuple[str, float], ...]

    @property
    def total_seconds(self) -> float:
        return sum(seconds for _, seconds in self.stage_timings)

    def format(self) -> str:
        total = self.total_seconds
        lines = [
            f"{self.title} performance profile",
            "-" * 76,
            f"{'Stage':<42}{'Time(s)':>12}{'Percent':>11}",
            "-" * 76,
        ]
        for stage, seconds in self.stage_timings:
            percent = 0.0 if total == 0 else seconds / total * 100.0
            lines.append(f"{stage:<42}{seconds:>12.6f}{percent:>10.2f}%")
        lines.extend(
            [
                "-" * 76,
                f"{'Total':<42}{total:>12.6f}{100.0 if total else 0.0:>10.2f}%",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class EncryptionResult:
    encrypted_image: np.ndarray
    metadata: EncryptionMetadata
    profile: EncryptionProfile


@dataclass(frozen=True)
class DecryptionResult:
    decrypted_image: np.ndarray
    profile: EncryptionProfile


class _ProfileRecorder:
    def __init__(self, title: str) -> None:
        self.title = title
        self._entries: list[tuple[str, float]] = []

    def record(self, stage: str, started_at: float) -> None:
        self._entries.append((stage, time.perf_counter() - started_at))

    def build(self) -> EncryptionProfile:
        return EncryptionProfile(self.title, tuple(self._entries))


def _load_rgb_image_array(image: str | Path | np.ndarray | pil_image.Image) -> np.ndarray:
    """Load a path, PIL image, or ndarray as a contiguous RGB uint8 array."""
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
        raise ValueError("3D image arrays must have at least one channel")

    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating) and image_array.size and np.nanmax(image_array) <= 1.0:
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_array)


def _image_to_bitplanes(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    height, width, _ = image.shape
    bitplanes = np.empty((24, height, width), dtype=np.uint8)
    plane = 0
    for channel in range(3):
        for bit in range(7, -1, -1):
            bitplanes[plane] = (image[..., channel] >> bit) & 1
            plane += 1
    return bitplanes


def _bitplanes_to_image(bitplanes: np.ndarray) -> np.ndarray:
    bitplanes = np.asarray(bitplanes, dtype=np.uint8)
    if bitplanes.ndim != 3 or bitplanes.shape[0] != 24:
        raise ValueError("bitplanes must have shape (24, H, W)")
    height, width = bitplanes.shape[1:]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    plane = 0
    for channel in range(3):
        for bit in range(7, -1, -1):
            image[..., channel] |= (bitplanes[plane] & 1) << bit
            plane += 1
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image[..., :3], dtype=np.float64) / 255.0
    rgb = _srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    xyz_matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float64,
    )
    xyz = np.tensordot(rgb, xyz_matrix.T, axes=1) / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f_xyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(f_xyz)
    lab[..., 0] = 116.0 * f_xyz[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return lab


def _gradient_magnitude(image: np.ndarray, method: str) -> np.ndarray:
    normalized = method.lower().strip()
    if normalized == "gray":
        row_grad, col_grad = np.gradient(_to_grayscale(image))
        return np.hypot(row_grad, col_grad)
    if normalized == "lab":
        lab = _rgb_to_lab(image)
        squared = np.zeros(lab.shape[:2], dtype=np.float64)
        for channel in range(3):
            row_grad, col_grad = np.gradient(lab[..., channel])
            squared += row_grad * row_grad + col_grad * col_grad
        return np.sqrt(squared)
    if normalized in {"di_zenzo", "dizenzo", "di-zenzo"}:
        rgb = np.asarray(image[..., :3], dtype=np.float64)
        g_xx = np.zeros(rgb.shape[:2], dtype=np.float64)
        g_yy = np.zeros(rgb.shape[:2], dtype=np.float64)
        g_xy = np.zeros(rgb.shape[:2], dtype=np.float64)
        for channel in range(3):
            row_grad, col_grad = np.gradient(rgb[..., channel])
            g_xx += col_grad * col_grad
            g_yy += row_grad * row_grad
            g_xy += col_grad * row_grad
        return np.sqrt(np.maximum(0.5 * (g_xx + g_yy + np.sqrt((g_xx - g_yy) ** 2 + 4.0 * g_xy**2)), 0.0))
    raise ValueError("gradient_method must be one of: 'gray', 'lab', 'di_zenzo'")


def _apply_dna_operation(left: np.ndarray, right: np.ndarray, operation: str) -> np.ndarray:
    operation = operation.lower().strip()
    if operation == "xor":
        return np.bitwise_xor(left, right).astype(np.uint8, copy=False)
    if operation == "add":
        return ((left.astype(np.uint16) + right.astype(np.uint16)) % 4).astype(np.uint8)
    if operation == "sub":
        return ((left.astype(np.int16) - right.astype(np.int16)) % 4).astype(np.uint8)
    raise ValueError("operation must be one of: 'xor', 'add', 'sub'")


def _operation_name_from_rule_id(rule_id: int) -> str:
    try:
        return DNA_OPERATION_RULE_ID_TO_NAME[int(rule_id)]
    except KeyError as error:
        raise ValueError("block operation rule ids must be integers in [1, 3]") from error


def _quantize_chaotic_sequence_12bit(sequence: np.ndarray, pixel_count: int) -> np.ndarray:
    """Quantize chaotic values to their low 12 bits without a broadcast bit tensor."""
    quantized = np.floor(
        np.asarray(sequence[:pixel_count], dtype=np.float64) * 10**10
    ).astype(np.uint64)
    np.bitwise_and(quantized, np.uint64(4095), out=quantized)
    return quantized.astype(np.uint16)


def _sequence_pair_to_dna_values(
    low_sequence: np.ndarray,
    high_sequence: np.ndarray,
    pixel_count: int,
) -> np.ndarray:
    """Return (pixel_count, 12) two-bit values from two chaotic sequences."""
    low_quantized = _quantize_chaotic_sequence_12bit(low_sequence, pixel_count)
    high_quantized = _quantize_chaotic_sequence_12bit(high_sequence, pixel_count)
    pair_values = DNA_12_BIT_LUT[high_quantized]
    np.left_shift(pair_values, np.uint8(1), out=pair_values)
    np.bitwise_or(pair_values, DNA_12_BIT_LUT[low_quantized], out=pair_values)
    return pair_values


def _apply_inverse_dna_operation(cipher: np.ndarray, key: np.ndarray, operation: str) -> np.ndarray:
    operation = operation.lower().strip()
    if operation == "xor":
        return np.bitwise_xor(cipher, key).astype(np.uint8, copy=False)
    if operation == "add":
        return ((cipher.astype(np.int16) - key.astype(np.int16)) % 4).astype(np.uint8)
    if operation == "sub":
        return ((cipher.astype(np.uint16) + key.astype(np.uint16)) % 4).astype(np.uint8)
    raise ValueError("operation must be one of: 'xor', 'add', 'sub'")


class Encrypter:
    """Encrypt RGB images and return ciphertext together with required metadata."""

    def __init__(
        self,
        config: EncryptionConfig | None = None,
        key_source: str | Path | np.ndarray | pil_image.Image | None = None,
    ) -> None:
        self.config = config or EncryptionConfig()
        self._cml: SalomoncouplingCML | None = None
        self._cml_source_array: np.ndarray | None = None
        self._cml_default_parameters: dict[str, float] | None = None
        if key_source is not None:
            self._initialize_cml(key_source)

    def set_cml(
        self,
        image: str | Path | np.ndarray | pil_image.Image | None = None,
        *,
        mu: float | None = None,
        v: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> dict[str, float]:
        """Temporarily replace selected CML parameters.

        Pass ``image`` when no CML has been initialized yet. The image-derived
        default parameters are saved before any replacements are applied.
        ``resume_cml()`` restores that snapshot. The returned mapping contains
        the effective parameter values after the update.
        """
        if image is not None:
            self._initialize_cml(image)
            self._cml_default_parameters = None
        if self._cml is None:
            raise RuntimeError("CML is not initialized; pass image=... to set_cml() first")

        if self._cml_default_parameters is None:
            self._cml_default_parameters = {
                name: float(getattr(self._cml, name))
                for name in ("mu", "v", "alpha", "beta")
            }

        updates = {"mu": mu, "v": v, "alpha": alpha, "beta": beta}
        validated_updates: dict[str, float] = {}
        for name, value in updates.items():
            if value is None:
                continue
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"{name} must be finite")
            validated_updates[name] = numeric_value
        for name, value in validated_updates.items():
            setattr(self._cml, name, value)

        return {
            name: float(getattr(self._cml, name))
            for name in ("mu", "v", "alpha", "beta")
        }

    def resume_cml(self) -> dict[str, float] | None:
        """Restore the CML parameters saved by ``set_cml()``.

        Encryption currently releases its CML after each call. In that case
        there is no live object to update, but clearing the saved snapshot makes
        the next image-dependent initialization use its unmodified defaults.
        """
        restored = self._cml_default_parameters
        if self._cml is not None and restored is not None:
            for name, value in restored.items():
                setattr(self._cml, name, value)
        self._cml_default_parameters = None
        return dict(restored) if restored is not None else None

    def _initialize_cml(self, image: str | Path | np.ndarray | pil_image.Image) -> None:
        """Create and warm up image-dependent CML outside encryption timing."""
        if isinstance(image, (str, Path)):
            key_source = image
            image_shape = self.load_image(image).shape
            self._cml_source_array = None
        else:
            key_source = _load_rgb_image_array(image)
            image_shape = key_source.shape
            self._cml_source_array = key_source.copy()
        self._cml = SalomoncouplingCML(L=image_shape[0] * image_shape[1], image_path=key_source)
        self._cml.generate_rdseq_fast(1)

    @staticmethod
    def load_image(image: str | Path | np.ndarray | pil_image.Image) -> np.ndarray:
        return _load_rgb_image_array(image)

    def adaptive_partition(self, image: np.ndarray) -> tuple[ImageBlock, ...]:
        """Recursively split high-gradient regions into smaller rectangular blocks."""
        gradient = _gradient_magnitude(image, self.config.gradient_method)
        threshold = self.config.gradient_threshold
        if threshold is None:
            threshold = 1.5 * float(np.mean(gradient))
        elif not np.isfinite(threshold):
            raise ValueError("gradient_threshold must be finite")

        height, width = gradient.shape
        blocks: list[ImageBlock] = []

        def split(row: int, col: int, block_height: int, block_width: int) -> None:
            mean_gradient = float(np.mean(gradient[row:row + block_height, col:col + block_width]))
            can_split = block_height > 1 and block_width > 1
            reached_minimum = block_height <= self.config.b_min or block_width <= self.config.b_min
            if reached_minimum or mean_gradient <= threshold or not can_split:
                blocks.append(ImageBlock(row, col, block_height, block_width, mean_gradient))
                return
            half_height = block_height // 2
            half_width = block_width // 2
            if half_height == 0 or half_width == 0:
                blocks.append(ImageBlock(row, col, block_height, block_width, mean_gradient))
                return
            for sub_row, sub_col, sub_height, sub_width in (
                (row, col, half_height, half_width),
                (row, col + half_width, half_height, block_width - half_width),
                (row + half_height, col, block_height - half_height, half_width),
                (row + half_height, col + half_width, block_height - half_height, block_width - half_width),
            ):
                split(sub_row, sub_col, sub_height, sub_width)

        for row in range(0, height, self.config.b_max):
            for col in range(0, width, self.config.b_max):
                split(row, col, min(self.config.b_max, height - row), min(self.config.b_max, width - col))
        return tuple(blocks)

    def generate_key_material(
        self,
        cml: SalomoncouplingCML,
        blocks: tuple[ImageBlock, ...],
        height: int,
        width: int,
        print_profile: bool = False,
    ) -> EncryptionKeyMaterial:
        """根据初始状态生成用于加密的秘钥矩阵/序列。"""
        recorder = _ProfileRecorder("Key material generation")
        started = time.perf_counter()
        B = len(blocks)
        N_OperationRule = B
        N_InnerPermutation = sum(DNA_PLANE_COUNT + block.height + block.width for block in blocks)
        N_GlobalPermutation = height * width
        N_GlobalKeyRule = height * width
        if not isinstance(cml, SalomoncouplingCML) or cml.L != height * width:
            raise ValueError("CML must be initialized and match the image dimensions")
        Matrix = cml.generate_rdseq_fast(7)  # shape: (7, L) array
        recorder.record("CML random matrix generation", started)

        started = time.perf_counter()
        required_values = 2 * B + N_InnerPermutation + N_OperationRule + N_GlobalPermutation + N_GlobalKeyRule
        # CHAOTIC SEQUENCES: use Matrix's final three rows in order.
        # No extra CML iteration is used for key-material expansion.
        Seq = np.concatenate((Matrix[-1, :], Matrix[-2, :], Matrix[-3, :]))
        if required_values > Seq.size:
            raise ValueError(
                f"Seq length {Seq.size} is insufficient for all key allocations; "
                f"required {required_values} values"
            )
        recorder.record("Sequence construction and capacity check", started)

        started = time.perf_counter()
        EncodeRule = (np.mod(Seq[:B] * 10**10, 8) + 1).astype(np.uint8)
        intra_permutations: list[BlockAxisPermutationIndices] = []
        cursor = B
        for block in blocks:
            plane_values = Seq[cursor:cursor + DNA_PLANE_COUNT]
            cursor += DNA_PLANE_COUNT
            row_values = Seq[cursor:cursor + block.height]
            cursor += block.height
            col_values = Seq[cursor:cursor + block.width]
            cursor += block.width
            intra_permutations.append(
                BlockAxisPermutationIndices(
                    plane_permutation=np.argsort(plane_values).astype(np.intp),
                    row_permutation=np.argsort(row_values).astype(np.intp),
                    col_permutation=np.argsort(col_values).astype(np.intp),
                )
            )
        recorder.record("Block encoding rules and axis permutations", started)

        started = time.perf_counter()
        shuffle_values = Seq[cursor:cursor + B]
        cursor += B
        grouped_indices: dict[tuple[int, int], list[int]] = {}
        for index, block in enumerate(blocks):
            grouped_indices.setdefault((block.height, block.width), []).append(index)
        block_shuffle_keys: list[BlockShuffleKey] = []
        shuffled_rule_ids = EncodeRule.copy()
        for shape, index_list in grouped_indices.items():
            targets = np.asarray(index_list, dtype=np.intp)
            sources = targets[np.argsort(shuffle_values[targets], kind="stable")]
            shuffled_rule_ids[targets] = EncodeRule[sources]
            block_shuffle_keys.append(BlockShuffleKey(shape, targets, sources))
        recorder.record("Same-size block shuffle keys", started)

        started = time.perf_counter()
        operation_rule_values = Seq[cursor:cursor + N_OperationRule]
        cursor += N_OperationRule
        block_operation_rule_ids = (
            np.mod(np.floor(operation_rule_values * 10**10), 3) + 1
        ).astype(np.uint8)
        recorder.record("Block DNA operation rules", started)

        started = time.perf_counter()
        # CHAOTIC SEQUENCE: ranks define the global spatial permutation used by
        # the ciphertext-feedback diffusion.
        global_permutation_values = Seq[cursor:cursor + N_GlobalPermutation]
        cursor += N_GlobalPermutation
        global_permutation = np.argsort(global_permutation_values, kind="stable").astype(np.intp)
        # CHAOTIC SEQUENCE: one DNA coding rule for every global-key pixel.
        global_key_rule_id = (
            np.mod(np.floor(Seq[cursor:cursor + N_GlobalKeyRule] * 10**10), 8) + 1
        ).astype(np.uint8)
        cursor += N_GlobalKeyRule

        if cursor != required_values:
            raise RuntimeError("key sequence allocation mismatch")
        recorder.record("Global permutation and DNA rules", started)

        started = time.perf_counter()
        pixel_count = height * width
        if Matrix.ndim != 2 or Matrix.shape[0] < 4 or Matrix.shape[1] < pixel_count:
            raise ValueError(
                f"Matrix must have shape (at least 4, at least {pixel_count}), got {Matrix.shape}"
            )

        # The paper's Matrix dimensions are one-based: dimensions 1 and 2 map
        # to Python rows 0 and 1. A small LUT expands only uint8 bit values,
        # avoiding the former 12 x H x W uint64 broadcast intermediates.
        block_key_values = _sequence_pair_to_dna_values(
            Matrix[0, :],
            Matrix[1, :],
            pixel_count,
        )
        block_rule_indices = np.empty((height, width), dtype=np.uint8)
        for block, rule_id in zip(blocks, shuffled_rule_ids, strict=True):
            block_rule_indices[block.row_slice, block.col_slice] = rule_id - 1
        full_block_key_matrix = DNA_RULE_TABLE[
            block_rule_indices.reshape(1, pixel_count),
            block_key_values.T,
        ].reshape(DNA_PLANE_COUNT, height, width)
        block_key_matrices = tuple(
            full_block_key_matrix[:, block.row_slice, block.col_slice]
            for block in blocks
        )
        del block_key_values, block_rule_indices

        if sum(matrix.size for matrix in block_key_matrices) != DNA_PLANE_COUNT * pixel_count:
            raise RuntimeError("block key matrix allocation does not cover the complete image")
        recorder.record("Block DNA key matrices", started)

        started = time.perf_counter()
        # Likewise, one-based Matrix dimensions 3 and 4 map to Python rows 2
        # and 3. Each spatial position uses its own global DNA key rule.
        global_key_values = _sequence_pair_to_dna_values(
            Matrix[2, :],
            Matrix[3, :],
            pixel_count,
        )
        global_key_matrix = DNA_RULE_TABLE[
            (global_key_rule_id - 1).reshape(1, pixel_count),
            global_key_values.T,
        ].reshape(DNA_PLANE_COUNT, height, width)
        del global_key_values
        # The first parallel diffusion group starts from C_{i-1}=0.
        global_initial_vector = np.zeros(DNA_PLANE_COUNT, dtype=np.uint8)
        recorder.record("Global DNA key matrix", started)

        started = time.perf_counter()
        key_material = EncryptionKeyMaterial(
            original_block_rule_ids=EncodeRule,
            shuffled_block_rule_ids=shuffled_rule_ids,
            intra_block_permutations=tuple(intra_permutations),
            block_shuffle_keys=tuple(block_shuffle_keys),
            block_key_matrices=block_key_matrices,
            block_operation_rule_ids=block_operation_rule_ids,
            global_permutation=global_permutation,
            global_key_matrix=global_key_matrix,
            global_initial_vector=global_initial_vector,
            global_key_rule_id=global_key_rule_id,
        )
        recorder.record("Key material packaging", started)
        if print_profile:
            print(recorder.build().format())
        return key_material

    def generate_key_material_old(
        self,
        blocks: tuple[ImageBlock, ...],
        height: int,
        width: int,
    ) -> EncryptionKeyMaterial:
        """Create and retain every random sequence/matrix used by this encryption."""
        # TODO(chaotic-rng): Replace this single NumPy pseudorandom source with a
        # chaotic-system key-stream generator.  Decryption uses the generated
        # arrays below from metadata, so its implementation will remain unchanged.
        rng = np.random.default_rng(self.config.seed)

        # PSEUDORANDOM SEQUENCE: one DNA coding-rule id per original image block.
        original_rule_ids = rng.integers(1, 9, size=len(blocks), dtype=np.uint8)

        intra_permutations: list[BlockAxisPermutationIndices] = []
        for block in blocks:
            # PSEUDORANDOM SEQUENCES: each axis receives an independent sequence.
            # Sorting them yields reversible plane, row, and column permutations.
            intra_permutations.append(
                BlockAxisPermutationIndices(
                    plane_permutation=np.argsort(rng.random(DNA_PLANE_COUNT)).astype(np.intp),
                    row_permutation=np.argsort(rng.random(block.height)).astype(np.intp),
                    col_permutation=np.argsort(rng.random(block.width)).astype(np.intp),
                )
            )

        grouped_indices: dict[tuple[int, int], list[int]] = {}
        for index, block in enumerate(blocks):
            grouped_indices.setdefault((block.height, block.width), []).append(index)
        shuffle_keys: list[BlockShuffleKey] = []
        shuffled_rule_ids = original_rule_ids.copy()
        for shape, index_list in grouped_indices.items():
            targets = np.asarray(index_list, dtype=np.intp)
            # PSEUDORANDOM SEQUENCE: source block order for each same-sized group.
            sources = rng.permutation(targets).astype(np.intp)
            shuffled_rule_ids[targets] = original_rule_ids[sources]
            shuffle_keys.append(BlockShuffleKey(shape, targets, sources))

        # PSEUDORANDOM SEQUENCE: one local DNA operation rule per shuffled block.
        block_operation_rule_ids = rng.integers(1, 4, size=len(blocks), dtype=np.uint8)

        block_key_matrices: list[np.ndarray] = []
        for block_index, block in enumerate(blocks):
            # PSEUDORANDOM MATRICES: two binary matrices per DNA plane are encoded
            # with this target block's rule to form its reversible local key matrix.
            msb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, block.height, block.width), dtype=np.uint8)
            lsb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, block.height, block.width), dtype=np.uint8)
            values = (msb << 1) | lsb
            block_key_matrices.append(DNA_RULE_TABLE[int(shuffled_rule_ids[block_index]) - 1][values])

        # PSEUDORANDOM SEQUENCE: global spatial order used by the feedback diffusion.
        global_permutation = rng.permutation(height * width).astype(np.intp)
        # PSEUDORANDOM SEQUENCE: one DNA coding rule per global-key pixel.
        global_key_rule_id = rng.integers(1, 9, size=height * width, dtype=np.uint8)
        # PSEUDORANDOM MATRICES: global two-bit key for every DNA plane and pixel.
        global_msb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, height, width), dtype=np.uint8)
        global_lsb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, height, width), dtype=np.uint8)
        global_key_matrix = DNA_RULE_TABLE[
            global_key_rule_id.reshape(height, width) - 1,
            (global_msb << 1) | global_lsb,
        ]
        # The first parallel group uses an all-zero C_{i-1}, as specified.
        # Keep the vector in metadata so decryption explicitly uses the same state.
        global_initial_vector = np.zeros(DNA_PLANE_COUNT, dtype=np.uint8)

        return EncryptionKeyMaterial(
            original_block_rule_ids=original_rule_ids,
            shuffled_block_rule_ids=shuffled_rule_ids,
            intra_block_permutations=tuple(intra_permutations),
            block_shuffle_keys=tuple(shuffle_keys),
            block_key_matrices=tuple(block_key_matrices),
            block_operation_rule_ids=block_operation_rule_ids,
            global_permutation=global_permutation,
            global_key_matrix=global_key_matrix,
            global_initial_vector=global_initial_vector,
            global_key_rule_id=global_key_rule_id,
        )

    @staticmethod
    def encode_to_dna(
        bitplanes: np.ndarray,
        blocks: Iterable[ImageBlock],
        block_rule_ids: np.ndarray,
    ) -> np.ndarray:
        """Encode all 24 RGB bitplanes into 12 DNA-code planes."""
        bitplanes = np.asarray(bitplanes, dtype=np.uint8)
        blocks = tuple(blocks)
        rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
        if bitplanes.ndim != 3 or bitplanes.shape[0] != 24:
            raise ValueError("bitplanes must have shape (24, H, W)")
        if rule_ids.shape != (len(blocks),):
            raise ValueError("block_rule_ids must have one rule for every block")
        height, width = bitplanes.shape[1:]
        dna = np.empty((DNA_PLANE_COUNT, height, width), dtype=np.uint8)
        for block_index, block in enumerate(blocks):
            values = (bitplanes[0:24:2, block.row_slice, block.col_slice] << 1) | bitplanes[1:24:2, block.row_slice, block.col_slice]
            dna[:, block.row_slice, block.col_slice] = DNA_RULE_TABLE[int(rule_ids[block_index]) - 1][values]
        return dna

    @staticmethod
    def decode_from_dna(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        block_rule_ids: np.ndarray,
    ) -> np.ndarray:
        """Convert 12 DNA-code planes into all 24 RGB bitplanes."""
        dna_matrix = np.asarray(dna_matrix, dtype=np.uint8)
        blocks = tuple(blocks)
        rule_ids = np.asarray(block_rule_ids, dtype=np.uint8)
        if dna_matrix.ndim != 3 or dna_matrix.shape[0] != DNA_PLANE_COUNT:
            raise ValueError(f"dna_matrix must have shape ({DNA_PLANE_COUNT}, H, W)")
        if np.any((dna_matrix < 0) | (dna_matrix > 3)):
            raise ValueError("dna_matrix must contain only values in [0, 3]")
        if rule_ids.shape != (len(blocks),):
            raise ValueError("block_rule_ids must have one rule for every block")
        height, width = dna_matrix.shape[1:]
        bitplanes = np.empty((24, height, width), dtype=np.uint8)
        for block_index, block in enumerate(blocks):
            values = DNA_RULE_TABLE_INV[int(rule_ids[block_index]) - 1][dna_matrix[:, block.row_slice, block.col_slice]]
            bitplanes[0:24:2, block.row_slice, block.col_slice] = values >> 1
            bitplanes[1:24:2, block.row_slice, block.col_slice] = values & 1
        return bitplanes

    @staticmethod
    def permute_within_blocks(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        permutations: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Forward 3D permutation: output_flat = input_flat[permutation]."""
        blocks = tuple(blocks)
        if len(permutations) != len(blocks):
            raise ValueError("one intra-block permutation is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block, permutation in zip(blocks, permutations, strict=True):
            source = dna_matrix[:, block.row_slice, block.col_slice].reshape(-1)
            if permutation.shape != source.shape:
                raise ValueError("intra-block permutation shape does not match its block")
            output[:, block.row_slice, block.col_slice] = source[permutation].reshape(DNA_PLANE_COUNT, block.height, block.width)
        return output

    @staticmethod
    def permute_within_blocks_V0(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        permutations: tuple[BlockAxisPermutationIndices, ...],
    ) -> np.ndarray:
        """Permute each block independently along its plane, row, and column axes."""
        blocks = tuple(blocks)
        if len(permutations) != len(blocks):
            raise ValueError("one axis permutation set is required for every block")

        source_dna = np.asarray(dna_matrix, dtype=np.uint8)
        output = source_dna.copy()
        for block, indices in zip(blocks, permutations, strict=True):
            if indices.plane_permutation.shape != (DNA_PLANE_COUNT,):
                raise ValueError("plane permutation must have length DNA_PLANE_COUNT")
            if indices.row_permutation.shape != (block.height,):
                raise ValueError("row permutation does not match its block")
            if indices.col_permutation.shape != (block.width,):
                raise ValueError("column permutation does not match its block")
            source_block = source_dna[:, block.row_slice, block.col_slice]
            output[:, block.row_slice, block.col_slice] = source_block[
                np.ix_(indices.plane_permutation, indices.row_permutation, indices.col_permutation)
            ]
        return output

    @staticmethod
    def shuffle_same_size_blocks(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        shuffle_keys: tuple[BlockShuffleKey, ...],
    ) -> np.ndarray:
        """Forward block shuffle: target block receives its selected source block."""
        blocks = tuple(blocks)
        source_dna = np.asarray(dna_matrix, dtype=np.uint8)
        output = source_dna.copy()
        for key in shuffle_keys:
            for target_index, source_index in zip(key.target_block_indices, key.source_block_indices, strict=True):
                target = blocks[int(target_index)]
                source = blocks[int(source_index)]
                output[:, target.row_slice, target.col_slice] = source_dna[:, source.row_slice, source.col_slice]
        return output

    def apply_block_diffusion(
        self,
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        key_matrices: tuple[np.ndarray, ...],
        operation_rule_ids: np.ndarray,
    ) -> np.ndarray:
        """Apply reversible per-block DNA operations selected by operation-rule ids."""
        blocks = tuple(blocks)
        operation_rule_ids = np.asarray(operation_rule_ids, dtype=np.uint8)
        if len(key_matrices) != len(blocks):
            raise ValueError("one block key matrix is required for every block")
        if operation_rule_ids.shape != (len(blocks),):
            raise ValueError("one block operation rule id is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block_index, (block, key_matrix) in enumerate(zip(blocks, key_matrices, strict=True)):
            expected_shape = (DNA_PLANE_COUNT, block.height, block.width)
            if key_matrix.shape != expected_shape:
                raise ValueError("block key matrix shape does not match its block")
            output[:, block.row_slice, block.col_slice] = _apply_dna_operation(
                dna_matrix[:, block.row_slice, block.col_slice],
                key_matrix,
                _operation_name_from_rule_id(int(operation_rule_ids[block_index])),
            )
        return output

    @staticmethod
    def apply_global_diffusion(
        dna_matrix: np.ndarray,
        key_material: EncryptionKeyMaterial,
        parallel_size: int = 1,
    ) -> np.ndarray:
        """Apply grouped, reversible ciphertext-feedback XOR in a random spatial order.

        Every group contains at most ``parallel_size`` spatial positions.  Its
        positions use the same predecessor vector and can therefore be computed
        synchronously.  The following group uses the previous group's final
        ciphertext vector as its predecessor.
        """
        dna_matrix = np.asarray(dna_matrix, dtype=np.uint8)
        if not isinstance(parallel_size, int) or parallel_size <= 0:
            raise ValueError("parallel_size must be a positive integer")
        _, height, width = dna_matrix.shape
        num_pixels = height * width
        permutation = key_material.global_permutation
        if permutation.shape != (num_pixels,):
            raise ValueError("global permutation does not match image dimensions")
        if key_material.global_key_matrix.shape != dna_matrix.shape:
            raise ValueError("global key matrix does not match DNA matrix dimensions")

        source_flat = dna_matrix.reshape(DNA_PLANE_COUNT, num_pixels)
        key_flat = key_material.global_key_matrix.reshape(DNA_PLANE_COUNT, num_pixels)
        plain_sequence = source_flat[:, permutation]
        key_sequence = key_flat[:, permutation]
        cipher_sequence = np.empty_like(plain_sequence)
        # The first group starts from the required all-zero C_{i-1} vector.
        previous_group_last_cipher = key_material.global_initial_vector.copy()
        for start in range(0, num_pixels, parallel_size):
            end = min(start + parallel_size, num_pixels)
            # All positions in [start, end) share one predecessor, so NumPy
            # evaluates the group's 12-channel DNA diffusion synchronously.
            cipher_sequence[:, start:end] = np.bitwise_xor(
                np.bitwise_xor(plain_sequence[:, start:end], previous_group_last_cipher[:, np.newaxis]),
                key_sequence[:, start:end],
            )
            # The final ciphertext vector feeds the next parallel group.
            previous_group_last_cipher = cipher_sequence[:, end - 1].copy()

        output_flat = np.empty_like(source_flat)
        output_flat[:, permutation] = cipher_sequence
        return output_flat.reshape(DNA_PLANE_COUNT, height, width)

    def encrypt(
        self,
        image: str | Path | np.ndarray | pil_image.Image,
        *,
        print_profile: bool = True,
    ) -> EncryptionResult:
        """Encrypt an image and return both ciphertext and the decryption metadata."""
        # CML setup and warmup are initialization work, excluded from encryption
        # and key-material timing profiles. Constructor key_source enables eager setup.
        if self._cml is None:
            self._initialize_cml(image)
        recorder = _ProfileRecorder("Encryption")

        started = time.perf_counter()
        image_array = self.load_image(image)
        recorder.record("Image loading / RGB conversion", started)

        started = time.perf_counter()
        blocks = self.adaptive_partition(image_array)
        recorder.record("Adaptive partition", started)

        started = time.perf_counter()
        key_material = self.generate_key_material(self._cml, blocks, *image_array.shape[:2])
        recorder.record("Key material generation", started)

        started = time.perf_counter()
        original_bitplanes = _image_to_bitplanes(image_array)
        dna_matrix = self.encode_to_dna(original_bitplanes, blocks, key_material.original_block_rule_ids)
        recorder.record("DNA encoding", started)


        started = time.perf_counter()
        permuted_dna = self.permute_within_blocks_V0(dna_matrix, blocks, key_material.intra_block_permutations)
        recorder.record("DNA intra-block permutation", started)

        started = time.perf_counter()
        shuffled_dna = self.shuffle_same_size_blocks(permuted_dna, blocks, key_material.block_shuffle_keys)
        recorder.record("Same-size block shuffle", started)
        # ttt
        started = time.perf_counter()
        locally_diffused_dna = self.apply_block_diffusion(
            shuffled_dna,
            blocks,
            key_material.block_key_matrices,
            key_material.block_operation_rule_ids,
        )
        recorder.record("Blockwise DNA diffusion", started)

        started = time.perf_counter()
        globally_diffused_dna = self.apply_global_diffusion(
            locally_diffused_dna,
            key_material,
            parallel_size=self.config.global_parallel_size,
        )
        recorder.record("Global DNA diffusion", started)

        started = time.perf_counter()
        encrypted_bitplanes = self.decode_from_dna(globally_diffused_dna, blocks, key_material.shuffled_block_rule_ids)
        recorder.record("DNA decoding to bitplanes", started)

        started = time.perf_counter()
        encrypted_image = _bitplanes_to_image(encrypted_bitplanes)
        recorder.record("Bitplanes to RGB image", started)

        metadata = EncryptionMetadata(
            version="encryption-v2",
            image_shape=tuple(int(value) for value in image_array.shape),
            blocks=blocks,
            config=self.config,
            key_material=key_material,
        )
        profile = recorder.build()
        if print_profile:
            print(profile.format())

        #每次加密后，都将self._cml 置为None，确保下一次加密时重新初始化CML
        return EncryptionResult(encrypted_image=encrypted_image, metadata=metadata, profile=profile)


class DeEncrypter:
    """Reverse every operation performed by :class:`Encrypter`."""

    def __init__(self, config: EncryptionConfig | None = None) -> None:
        self.config = config

    @staticmethod
    def load_image(image: str | Path | np.ndarray | pil_image.Image) -> np.ndarray:
        return _load_rgb_image_array(image)

    @staticmethod
    def encode_to_dna(
        bitplanes: np.ndarray,
        blocks: Iterable[ImageBlock],
        block_rule_ids: np.ndarray,
    ) -> np.ndarray:
        return Encrypter.encode_to_dna(bitplanes, blocks, block_rule_ids)

    @staticmethod
    def invert_global_diffusion(
        dna_matrix: np.ndarray,
        key_material: EncryptionKeyMaterial,
        parallel_size: int = 1,
    ) -> np.ndarray:
        """Inverse of grouped ciphertext-feedback global XOR diffusion."""
        dna_matrix = np.asarray(dna_matrix, dtype=np.uint8)
        if not isinstance(parallel_size, int) or parallel_size <= 0:
            raise ValueError("parallel_size must be a positive integer")
        _, height, width = dna_matrix.shape
        num_pixels = height * width
        permutation = key_material.global_permutation
        if permutation.shape != (num_pixels,):
            raise ValueError("global permutation does not match image dimensions")
        if key_material.global_key_matrix.shape != dna_matrix.shape:
            raise ValueError("global key matrix does not match DNA matrix dimensions")

        cipher_flat = dna_matrix.reshape(DNA_PLANE_COUNT, num_pixels)
        key_flat = key_material.global_key_matrix.reshape(DNA_PLANE_COUNT, num_pixels)
        cipher_sequence = cipher_flat[:, permutation]
        key_sequence = key_flat[:, permutation]
        plain_sequence = np.empty_like(cipher_sequence)
        previous_group_last_cipher = key_material.global_initial_vector.copy()
        for start in range(0, num_pixels, parallel_size):
            end = min(start + parallel_size, num_pixels)
            plain_sequence[:, start:end] = np.bitwise_xor(
                np.bitwise_xor(cipher_sequence[:, start:end], previous_group_last_cipher[:, np.newaxis]),
                key_sequence[:, start:end],
            )
            previous_group_last_cipher = cipher_sequence[:, end - 1].copy()

        output_flat = np.empty_like(cipher_flat)
        output_flat[:, permutation] = plain_sequence
        return output_flat.reshape(DNA_PLANE_COUNT, height, width)

    @staticmethod
    def invert_block_diffusion(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        key_matrices: tuple[np.ndarray, ...],
        operation_rule_ids: np.ndarray,
    ) -> np.ndarray:
        """Inverse of Encrypter.apply_block_diffusion."""
        blocks = tuple(blocks)
        operation_rule_ids = np.asarray(operation_rule_ids, dtype=np.uint8)
        if len(key_matrices) != len(blocks):
            raise ValueError("one block key matrix is required for every block")
        if operation_rule_ids.shape != (len(blocks),):
            raise ValueError("one block operation rule id is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block_index, (block, key_matrix) in enumerate(zip(blocks, key_matrices, strict=True)):
            expected_shape = (DNA_PLANE_COUNT, block.height, block.width)
            if key_matrix.shape != expected_shape:
                raise ValueError("block key matrix shape does not match its block")
            output[:, block.row_slice, block.col_slice] = _apply_inverse_dna_operation(
                dna_matrix[:, block.row_slice, block.col_slice],
                key_matrix,
                _operation_name_from_rule_id(int(operation_rule_ids[block_index])),
            )
        return output

    @staticmethod
    def invert_same_size_block_shuffle(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        shuffle_keys: tuple[BlockShuffleKey, ...],
    ) -> np.ndarray:
        """Inverse shuffle: each original source receives data from its target."""
        blocks = tuple(blocks)
        shuffled_dna = np.asarray(dna_matrix, dtype=np.uint8)
        output = shuffled_dna.copy()
        for key in shuffle_keys:
            for target_index, source_index in zip(key.target_block_indices, key.source_block_indices, strict=True):
                target = blocks[int(target_index)]
                source = blocks[int(source_index)]
                output[:, source.row_slice, source.col_slice] = shuffled_dna[:, target.row_slice, target.col_slice]
        return output

    @staticmethod
    def invert_intra_block_permutation(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        permutations: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Inverse of output_flat = input_flat[permutation]."""
        blocks = tuple(blocks)
        if len(permutations) != len(blocks):
            raise ValueError("one intra-block permutation is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block, permutation in zip(blocks, permutations, strict=True):
            source = dna_matrix[:, block.row_slice, block.col_slice].reshape(-1)
            if permutation.shape != source.shape:
                raise ValueError("intra-block permutation shape does not match its block")
            inverse_permutation = np.argsort(permutation)
            output[:, block.row_slice, block.col_slice] = source[inverse_permutation].reshape(
                DNA_PLANE_COUNT, block.height, block.width
            )
        return output

    @staticmethod
    def invert_permute_within_blocks_V0(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        permutations: tuple[BlockAxisPermutationIndices, ...],
    ) -> np.ndarray:
        """Inverse of :meth:`Encrypter.permute_within_blocks_V0`."""
        blocks = tuple(blocks)
        if len(permutations) != len(blocks):
            raise ValueError("one axis permutation set is required for every block")

        permuted_dna = np.asarray(dna_matrix, dtype=np.uint8)
        output = permuted_dna.copy()
        for block, indices in zip(blocks, permutations, strict=True):
            if indices.plane_permutation.shape != (DNA_PLANE_COUNT,):
                raise ValueError("plane permutation must have length DNA_PLANE_COUNT")
            if indices.row_permutation.shape != (block.height,):
                raise ValueError("row permutation does not match its block")
            if indices.col_permutation.shape != (block.width,):
                raise ValueError("column permutation does not match its block")
            inverse_plane = np.argsort(indices.plane_permutation)
            inverse_row = np.argsort(indices.row_permutation)
            inverse_col = np.argsort(indices.col_permutation)
            source_block = permuted_dna[:, block.row_slice, block.col_slice]
            output[:, block.row_slice, block.col_slice] = source_block[
                np.ix_(inverse_plane, inverse_row, inverse_col)
            ]
        return output

    @staticmethod
    def decode_from_dna(
        dna_matrix: np.ndarray,
        blocks: Iterable[ImageBlock],
        block_rule_ids: np.ndarray,
    ) -> np.ndarray:
        return Encrypter.decode_from_dna(dna_matrix, blocks, block_rule_ids)

    def decrypt(
        self,
        encrypted_image: str | Path | np.ndarray | pil_image.Image,
        metadata: EncryptionMetadata,
        *,
        print_profile: bool = True,
    ) -> DecryptionResult:
        """Decrypt an image using the exact random material retained at encryption."""
        if metadata.version != "encryption-v2":
            raise ValueError(f"unsupported encryption metadata version: {metadata.version}")
        if self.config is not None and self.config != metadata.config:
            raise ValueError("DeEncrypter config does not match the encryption metadata")

        recorder = _ProfileRecorder("Decryption")
        started = time.perf_counter()
        cipher_array = self.load_image(encrypted_image)
        recorder.record("Image loading / RGB conversion", started)
        if tuple(cipher_array.shape) != metadata.image_shape:
            raise ValueError("ciphertext dimensions do not match encryption metadata")

        started = time.perf_counter()
        # No new randomness is generated here.  These are the exact pseudorandom
        # sequences/matrices created by Encrypter.generate_key_material().
        key_material = metadata.key_material
        blocks = metadata.blocks
        self._validate_key_material(key_material, blocks, cipher_array.shape[:2])
        recorder.record("Encryption random material reuse", started)

        started = time.perf_counter()
        cipher_bitplanes = _image_to_bitplanes(cipher_array)
        cipher_dna = self.encode_to_dna(cipher_bitplanes, blocks, key_material.shuffled_block_rule_ids)
        recorder.record("DNA encoding", started)

        started = time.perf_counter()
        locally_diffused_dna = self.invert_global_diffusion(
            cipher_dna,
            key_material,
            parallel_size=metadata.config.global_parallel_size,
        )
        recorder.record("Inverse global DNA diffusion", started)

        started = time.perf_counter()
        shuffled_dna = self.invert_block_diffusion(
            locally_diffused_dna,
            blocks,
            key_material.block_key_matrices,
            key_material.block_operation_rule_ids,
        )
        recorder.record("Inverse blockwise DNA diffusion", started)

        started = time.perf_counter()
        permuted_dna = self.invert_same_size_block_shuffle(shuffled_dna, blocks, key_material.block_shuffle_keys)
        recorder.record("Inverse same-size block shuffle", started)

        # ttt
        started = time.perf_counter()
        original_dna = self.invert_permute_within_blocks_V0(
            permuted_dna,
            blocks,
            key_material.intra_block_permutations,
        )
        recorder.record("Inverse DNA intra-block permutation", started)

        started = time.perf_counter()
        original_bitplanes = self.decode_from_dna(original_dna, blocks, key_material.original_block_rule_ids)
        recorder.record("DNA decoding to bitplanes", started)

        started = time.perf_counter()
        decrypted_image = _bitplanes_to_image(original_bitplanes)
        recorder.record("Bitplanes to RGB image", started)

        profile = recorder.build()
        if print_profile:
            print(profile.format())
        return DecryptionResult(decrypted_image=decrypted_image, profile=profile)

    def Decrypt_V2(
        self,
        encrypted_image: str | Path | np.ndarray | pil_image.Image,
        encrypter: Encrypter,
        metadata: EncryptionMetadata,
        *,
        print_profile: bool = True,
    ) -> DecryptionResult:
        """Decrypt by regenerating key material from a prepared Encrypter CML.

        ``metadata`` supplies public image shape, adaptive block layout, and
        encryption configuration. Its original ``key_material`` is ignored.
        """
        if not isinstance(encrypter, Encrypter):
            raise TypeError("encrypter must be an Encrypter instance")
        if encrypter._cml is None:
            raise ValueError("encrypter._cml is unavailable; initialize it with the encryption CML state")
        if metadata.version != "encryption-v2":
            raise ValueError(f"unsupported encryption metadata version: {metadata.version}")
        if self.config is not None and self.config != metadata.config:
            raise ValueError("DeEncrypter config does not match the encryption metadata")

        cipher_array = self.load_image(encrypted_image)
        if tuple(cipher_array.shape) != metadata.image_shape:
            raise ValueError("ciphertext dimensions do not match encryption metadata")

        # generate_rdseq_fast() restarts from cml.x0, reconstructing encryption
        # key material without reading metadata.key_material.
        regenerated_key_material = encrypter.generate_key_material(
            encrypter._cml,
            metadata.blocks,
            *cipher_array.shape[:2],
            print_profile=print_profile,
        )
        regenerated_metadata = replace(metadata, key_material=regenerated_key_material)
        return self.decrypt(cipher_array, regenerated_metadata, print_profile=print_profile)

    @staticmethod
    def _validate_key_material(
        key_material: EncryptionKeyMaterial,
        blocks: tuple[ImageBlock, ...],
        image_shape: tuple[int, int],
    ) -> None:
        height, width = image_shape
        if key_material.original_block_rule_ids.shape != (len(blocks),):
            raise ValueError("original block-rule metadata is invalid")
        if key_material.shuffled_block_rule_ids.shape != (len(blocks),):
            raise ValueError("shuffled block-rule metadata is invalid")
        if len(key_material.intra_block_permutations) != len(blocks):
            raise ValueError("intra-block permutation metadata is invalid")
        for block, indices in zip(blocks, key_material.intra_block_permutations, strict=True):
            if not isinstance(indices, BlockAxisPermutationIndices):
                raise ValueError("intra-block axis permutation metadata is invalid")
            if indices.plane_permutation.shape != (DNA_PLANE_COUNT,):
                raise ValueError("intra-block plane permutation metadata is invalid")
            if indices.row_permutation.shape != (block.height,):
                raise ValueError("intra-block row permutation metadata is invalid")
            if indices.col_permutation.shape != (block.width,):
                raise ValueError("intra-block column permutation metadata is invalid")
        if len(key_material.block_key_matrices) != len(blocks):
            raise ValueError("block diffusion-key metadata is invalid")
        if key_material.block_operation_rule_ids.shape != (len(blocks),):
            raise ValueError("block operation-rule metadata is invalid")
        if np.any((key_material.block_operation_rule_ids < 1) | (key_material.block_operation_rule_ids > 3)):
            raise ValueError("block operation-rule metadata must contain values in [1, 3]")
        if key_material.global_permutation.shape != (height * width,):
            raise ValueError("global permutation metadata is invalid")
        if key_material.global_key_matrix.shape != (DNA_PLANE_COUNT, height, width):
            raise ValueError("global key-matrix metadata is invalid")
        if key_material.global_initial_vector.shape != (DNA_PLANE_COUNT,):
            raise ValueError("global initial-vector metadata is invalid")
        if key_material.global_key_rule_id.shape != (height * width,):
            raise ValueError("global key-rule metadata must have one rule per pixel")
        if np.any((key_material.global_key_rule_id < 1) | (key_material.global_key_rule_id > 8)):
            raise ValueError("global key-rule metadata must contain values in [1, 8]")


def plot_image_comparison(
    original_image: np.ndarray,
    encrypted_image: np.ndarray,
    decrypted_image: np.ndarray,
    save_path: str | Path,
    *,
    show: bool = True,
) -> Path:
    """Plot original, encrypted, and decrypted images side by side."""
    images = (
        (np.asarray(original_image), "Original image"),
        (np.asarray(encrypted_image), "Encrypted image"),
        (np.asarray(decrypted_image), "Decrypted image"),
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (image, title) in zip(axes, images, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Image Encryption / Decryption Comparison")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    # Non-interactive CI/headless backends can save the figure but cannot display it.
    backend = str(plt.get_backend()).lower()
    if show and backend not in {"agg", "pdf", "ps", "svg", "cairo"}:
        plt.show()
    plt.close(fig)
    return save_path


def demo(
    image_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    show_plot: bool = True,
) -> tuple[EncryptionResult, DecryptionResult, Path, Path]:
    """Run a small end-to-end encryption/decryption example.

    The demo prints both performance profiles, saves the ciphertext and the
    recovered image, and verifies that recovery is pixel-exact.
    """
    module_dir = Path(__file__).resolve().parent
    if image_path is None:
        image_path = module_dir.parent / "images" / "img3.png"
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"demo image does not exist: {image_path}")

    if output_dir is None:
        output_dir = module_dir / "outputs" / "encryption_demo"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo] input image: {image_path}")
    print_profile = False
    config = EncryptionConfig(seed=2026, b_max=64, b_min=16, block_operation="xor", global_parallel_size=64)
    encrypter = Encrypter(config)
    encryption_result = encrypter.encrypt(image_path, print_profile=True)

    encrypted_path = output_dir / "encrypted.png"
    pil_image.fromarray(encryption_result.encrypted_image).save(encrypted_path)

    decrypter = DeEncrypter()
    decryption_result = decrypter.decrypt(
        encryption_result.encrypted_image,
        encryption_result.metadata,
        print_profile=print_profile,
    )
    decrypted_path = output_dir / "decrypted.png"
    pil_image.fromarray(decryption_result.decrypted_image).save(decrypted_path)

    original_image = encrypter.load_image(image_path)
    comparison_path = output_dir / "comparison.png"
    plot_image_comparison(
        original_image,
        encryption_result.encrypted_image,
        decryption_result.decrypted_image,
        comparison_path,
        show=show_plot,
    )
    restored_exactly = bool(np.array_equal(original_image, decryption_result.decrypted_image))
    print(f"[demo] adaptive blocks: {len(encryption_result.metadata.blocks)}")
    print(f"[demo] global parallel size: {config.global_parallel_size}")
    print(f"[demo] decrypted exactly: {restored_exactly}")
    print(f"[demo] encrypted image: {encrypted_path}")
    print(f"[demo] decrypted image: {decrypted_path}")
    print(f"[demo] comparison plot: {comparison_path}")
    if not restored_exactly:
        raise RuntimeError("demo decryption did not reproduce the original image")

    return encryption_result, decryption_result, encrypted_path, decrypted_path


def main() -> int:
    """Main entry point used when running ``python Encryption.py``."""
    demo()
    return 0





__all__ = [
    "BlockAxisPermutationIndices",
    "BlockShuffleKey",
    "DeEncrypter",
    "DecryptionResult",
    "Encrypter",
    "EncryptionConfig",
    "EncryptionKeyMaterial",
    "EncryptionMetadata",
    "EncryptionProfile",
    "EncryptionResult",
    "ImageBlock",
    "demo",
    "main",
    "plot_image_comparison",
]


if __name__ == "__main__":
    raise SystemExit(main())
