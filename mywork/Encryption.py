"""Reversible adaptive DNA image encryption.

The module deliberately keeps all encryption random material in
``EncryptionMetadata``.  The current source is NumPy's pseudorandom generator;
the marked locations are the single replacement points for a future chaotic
key-stream generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pil_image


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
class EncryptionKeyMaterial:
    """All key and state material required by both encryption and decryption."""

    original_block_rule_ids: np.ndarray
    shuffled_block_rule_ids: np.ndarray
    intra_block_permutations: tuple[np.ndarray, ...]
    block_shuffle_keys: tuple[BlockShuffleKey, ...]
    block_key_matrices: tuple[np.ndarray, ...]
    global_permutation: np.ndarray
    global_key_matrix: np.ndarray
    global_initial_vector: np.ndarray
    global_key_rule_id: int


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

    def __init__(self, config: EncryptionConfig | None = None) -> None:
        self.config = config or EncryptionConfig()

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

        intra_permutations: list[np.ndarray] = []
        for block in blocks:
            # PSEUDORANDOM SEQUENCE: ranks define a reversible 3D permutation of
            # the 12 DNA planes and all pixel positions inside this block.
            length = DNA_PLANE_COUNT * block.height * block.width
            intra_permutations.append(np.argsort(rng.random(length)).astype(np.intp))

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
        # PSEUDORANDOM VALUE: rule used to encode the global DNA key material.
        global_key_rule_id = int(rng.integers(1, 9, dtype=np.uint8))
        # PSEUDORANDOM MATRICES: global two-bit key for every DNA plane and pixel.
        global_msb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, height, width), dtype=np.uint8)
        global_lsb = rng.integers(0, 2, size=(DNA_PLANE_COUNT, height, width), dtype=np.uint8)
        global_key_matrix = DNA_RULE_TABLE[global_key_rule_id - 1][(global_msb << 1) | global_lsb]
        # The first parallel group uses an all-zero C_{i-1}, as specified.
        # Keep the vector in metadata so decryption explicitly uses the same state.
        global_initial_vector = np.zeros(DNA_PLANE_COUNT, dtype=np.uint8)

        return EncryptionKeyMaterial(
            original_block_rule_ids=original_rule_ids,
            shuffled_block_rule_ids=shuffled_rule_ids,
            intra_block_permutations=tuple(intra_permutations),
            block_shuffle_keys=tuple(shuffle_keys),
            block_key_matrices=tuple(block_key_matrices),
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
    ) -> np.ndarray:
        """Apply reversible per-element DNA key mixing within each block."""
        blocks = tuple(blocks)
        if len(key_matrices) != len(blocks):
            raise ValueError("one block key matrix is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block, key_matrix in zip(blocks, key_matrices, strict=True):
            expected_shape = (DNA_PLANE_COUNT, block.height, block.width)
            if key_matrix.shape != expected_shape:
                raise ValueError("block key matrix shape does not match its block")
            output[:, block.row_slice, block.col_slice] = _apply_dna_operation(
                dna_matrix[:, block.row_slice, block.col_slice], key_matrix, self.config.block_operation
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
        recorder = _ProfileRecorder("Encryption")

        started = time.perf_counter()
        image_array = self.load_image(image)
        recorder.record("Image loading / RGB conversion", started)

        started = time.perf_counter()
        blocks = self.adaptive_partition(image_array)
        recorder.record("Adaptive partition", started)

        started = time.perf_counter()
        key_material = self.generate_key_material(blocks, *image_array.shape[:2])
        recorder.record("Pseudorandom key material generation", started)

        started = time.perf_counter()
        original_bitplanes = _image_to_bitplanes(image_array)
        dna_matrix = self.encode_to_dna(original_bitplanes, blocks, key_material.original_block_rule_ids)
        recorder.record("DNA encoding", started)

        started = time.perf_counter()
        permuted_dna = self.permute_within_blocks(dna_matrix, blocks, key_material.intra_block_permutations)
        recorder.record("DNA intra-block permutation", started)

        started = time.perf_counter()
        shuffled_dna = self.shuffle_same_size_blocks(permuted_dna, blocks, key_material.block_shuffle_keys)
        recorder.record("Same-size block shuffle", started)

        started = time.perf_counter()
        locally_diffused_dna = self.apply_block_diffusion(shuffled_dna, blocks, key_material.block_key_matrices)
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
            version="encryption-v1",
            image_shape=tuple(int(value) for value in image_array.shape),
            blocks=blocks,
            config=self.config,
            key_material=key_material,
        )
        profile = recorder.build()
        if print_profile:
            print(profile.format())
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
        operation: str,
    ) -> np.ndarray:
        """Inverse of Encrypter.apply_block_diffusion."""
        blocks = tuple(blocks)
        if len(key_matrices) != len(blocks):
            raise ValueError("one block key matrix is required for every block")
        output = np.asarray(dna_matrix, dtype=np.uint8).copy()
        for block, key_matrix in zip(blocks, key_matrices, strict=True):
            expected_shape = (DNA_PLANE_COUNT, block.height, block.width)
            if key_matrix.shape != expected_shape:
                raise ValueError("block key matrix shape does not match its block")
            output[:, block.row_slice, block.col_slice] = _apply_inverse_dna_operation(
                dna_matrix[:, block.row_slice, block.col_slice], key_matrix, operation
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
        if metadata.version != "encryption-v1":
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
            locally_diffused_dna, blocks, key_material.block_key_matrices, metadata.config.block_operation
        )
        recorder.record("Inverse blockwise DNA diffusion", started)

        started = time.perf_counter()
        permuted_dna = self.invert_same_size_block_shuffle(shuffled_dna, blocks, key_material.block_shuffle_keys)
        recorder.record("Inverse same-size block shuffle", started)

        started = time.perf_counter()
        original_dna = self.invert_intra_block_permutation(permuted_dna, blocks, key_material.intra_block_permutations)
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
        if len(key_material.block_key_matrices) != len(blocks):
            raise ValueError("block diffusion-key metadata is invalid")
        if key_material.global_permutation.shape != (height * width,):
            raise ValueError("global permutation metadata is invalid")
        if key_material.global_key_matrix.shape != (DNA_PLANE_COUNT, height, width):
            raise ValueError("global key-matrix metadata is invalid")
        if key_material.global_initial_vector.shape != (DNA_PLANE_COUNT,):
            raise ValueError("global initial-vector metadata is invalid")


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
    config = EncryptionConfig(seed=2026, b_max=64, b_min=12, block_operation="xor", global_parallel_size=64)
    encrypter = Encrypter(config)
    encryption_result = encrypter.encrypt(image_path, print_profile=True)

    encrypted_path = output_dir / "encrypted.png"
    pil_image.fromarray(encryption_result.encrypted_image).save(encrypted_path)

    decrypter = DeEncrypter()
    decryption_result = decrypter.decrypt(
        encryption_result.encrypted_image,
        encryption_result.metadata,
        print_profile=True,
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
