from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional convenience dependency
    Image = None


ALLOWED_LEU = (1, 2, 4, 8)
_BIT_WEIGHTS_DESC = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
_BIT_WEIGHTS_ASC = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint16)


@dataclass(frozen=True)
class EFIEDebugState:
    i1_shape: tuple[int, int]
    i2_shape: tuple[int, int]
    grouped_shape: tuple[int, int]
    i3_shape: tuple[int, int]
    i4_shape: tuple[int, int]
    i6_shape: tuple[int, int]
    seed: int


def encrypt_image(image: np.ndarray, leu: int) -> np.ndarray:
    """Encrypt a 24-bit RGB image with the EFIE procedure from the paper."""
    ciphertext, _ = _encrypt_core(image=image, leu=leu, key_image=image, return_debug=False)
    return ciphertext


def decrypt_image(ciphertext: np.ndarray, key_image: np.ndarray, leu: int) -> np.ndarray:
    """Decrypt a ciphertext image with the EFIE inverse procedure."""
    _validate_rgb_image(ciphertext, "ciphertext")
    _validate_rgb_image(key_image, "key_image")
    _validate_same_shape(ciphertext, key_image, "ciphertext", "key_image")
    _validate_leu(leu)

    seed = _generate_seed_from_key_image(key_image)
    rng = random.Random(seed)

    i7 = _channel_separation(ciphertext)
    i8 = _decimal_matrix_to_binary_matrix(i7)

    grouped_i8 = _group_binary_units(i8, leu)
    wr, wc = _build_reverse_lookup(grouped_i8.shape, rng)
    restored_grouped = _inverse_shuffle_grouped_matrix(grouped_i8, wr, wc)
    i9 = _ungroup_binary_units(restored_grouped, leu, ciphertext.shape[0], ciphertext.shape[1])

    i10 = _binary_matrix_to_decimal_matrix(i9)
    plaintext = _channel_merging(i10, ciphertext.shape[0], ciphertext.shape[1])
    return plaintext


def _encrypt_core(
    image: np.ndarray,
    leu: int,
    key_image: np.ndarray,
    return_debug: bool,
) -> tuple[np.ndarray, EFIEDebugState | None]:
    _validate_rgb_image(image, "image")
    _validate_rgb_image(key_image, "key_image")
    _validate_same_shape(image, key_image, "image", "key_image")
    _validate_leu(leu)

    seed = _generate_seed_from_key_image(key_image)
    rng = random.Random(seed)

    i1 = _channel_separation(image)
    i2 = _decimal_matrix_to_binary_matrix(i1)
    grouped_i2 = _group_binary_units(i2, leu)
    shuffled_grouped = _shuffle_grouped_matrix(grouped_i2, rng)
    i3 = _ungroup_binary_units(shuffled_grouped, leu, image.shape[0], image.shape[1])
    i6, i4_shape = _binary_matrix_to_decimal_matrix_with_i4_shape(i3)
    ciphertext = _channel_merging(i6, image.shape[0], image.shape[1])

    debug_state = None
    if return_debug:
        debug_state = EFIEDebugState(
            i1_shape=i1.shape,
            i2_shape=i2.shape,
            grouped_shape=grouped_i2.shape,
            i3_shape=i3.shape,
            i4_shape=i4_shape,
            i6_shape=i6.shape,
            seed=seed,
        )
    return ciphertext, debug_state


def _validate_rgb_image(image: np.ndarray, name: str) -> None:
    if image is None:
        raise ValueError(f"{name} must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (M, N, 3) for a 24-bit RGB image.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} must have positive height and width.")


def _validate_same_shape(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same shape.")


def _validate_leu(leu: int) -> None:
    if leu not in ALLOWED_LEU:
        raise ValueError(f"leu must be one of {ALLOWED_LEU}.")


def _channel_separation(image: np.ndarray) -> np.ndarray:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return np.concatenate((r, g, b), axis=1)


def _channel_merging(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    if matrix.shape != (height, width * 3):
        raise ValueError("Input matrix shape does not match channel merging requirements.")
    r = matrix[:, 0:width]
    g = matrix[:, width : 2 * width]
    b = matrix[:, 2 * width : 3 * width]
    return np.stack((r, g, b), axis=2).astype(np.uint8, copy=False)


def _decimal_matrix_to_binary_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_u16 = matrix.astype(np.uint16, copy=False)
    remainder = matrix_u16.copy()
    bit_planes = []
    for weight in _BIT_WEIGHTS_DESC:
        plane = (remainder >= weight).astype(np.uint8)
        bit_planes.append(plane)
        remainder = remainder - plane.astype(np.uint16) * weight

    # Algorithm 1 writes to B[7 - i], so the stored order is from low bit to high bit.
    little_endian_planes = bit_planes[::-1]
    return np.concatenate(little_endian_planes, axis=1)


def _binary_matrix_to_decimal_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    decimal_matrix, _ = _binary_matrix_to_decimal_matrix_with_i4_shape(binary_matrix)
    return decimal_matrix


def _binary_matrix_to_decimal_matrix_with_i4_shape(binary_matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width_times_24 = binary_matrix.shape
    if width_times_24 % 24 != 0:
        raise ValueError("Binary matrix width must be divisible by 24.")

    width = width_times_24 // 24
    i4 = binary_matrix.T.reshape((8, height * width * 3), order="F").T
    decimals = (i4.astype(np.uint16) * _BIT_WEIGHTS_ASC).sum(axis=1).astype(np.uint8)
    i6 = decimals.reshape((width * 3, height), order="F").T
    return i6, i4.shape


def _group_binary_units(binary_matrix: np.ndarray, leu: int) -> np.ndarray:
    if leu == 1:
        return binary_matrix.copy()
    if binary_matrix.shape[1] % leu != 0:
        raise ValueError("Binary matrix width must be divisible by leu.")

    height, width = binary_matrix.shape
    grouped = binary_matrix.reshape(height, width // leu, leu)
    weights = (1 << np.arange(leu, dtype=np.uint16)).reshape(1, 1, leu)
    return (grouped.astype(np.uint16) * weights).sum(axis=2).astype(np.uint8)


def _ungroup_binary_units(grouped_matrix: np.ndarray, leu: int, height: int, width: int) -> np.ndarray:
    if leu == 1:
        result = grouped_matrix.astype(np.uint8, copy=True)
    else:
        values = grouped_matrix.astype(np.uint16, copy=False)
        bits = np.zeros((height, grouped_matrix.shape[1], leu), dtype=np.uint8)
        remainder = values.copy()
        desc_weights = np.array([1 << bit for bit in range(leu - 1, -1, -1)], dtype=np.uint16)
        for idx, weight in enumerate(desc_weights):
            plane = (remainder >= weight).astype(np.uint8)
            bits[:, :, leu - 1 - idx] = plane
            remainder = remainder - plane.astype(np.uint16) * weight
        result = bits.reshape(height, grouped_matrix.shape[1] * leu)

    expected_shape = (height, width * 24)
    if result.shape != expected_shape:
        raise ValueError(f"Ungrouped binary matrix must have shape {expected_shape}, got {result.shape}.")
    return result


def _shuffle_grouped_matrix(grouped_matrix: np.ndarray, rng: random.Random) -> np.ndarray:
    flat = grouped_matrix.reshape(-1).copy()
    total = flat.size
    for idx in range(total - 1, -1, -1):
        swap_idx = rng.randint(0, idx)
        flat[idx], flat[swap_idx] = flat[swap_idx], flat[idx]
    return flat.reshape(grouped_matrix.shape)


def _build_reverse_lookup(shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    total = rows * cols
    wr = np.zeros(shape, dtype=np.int64)
    wc = np.zeros(shape, dtype=np.int64)
    for idx in range(total - 1, -1, -1):
        swap_idx = rng.randint(0, idx)
        row = idx // cols
        col = idx % cols
        wr[row, col] = swap_idx // cols
        wc[row, col] = swap_idx % cols
    return wr, wc


def _inverse_shuffle_grouped_matrix(
    grouped_matrix: np.ndarray, wr: np.ndarray, wc: np.ndarray
) -> np.ndarray:
    rows, cols = grouped_matrix.shape
    restored = grouped_matrix.copy()
    for row in range(rows):
        for col in range(cols):
            swap_row = int(wr[row, col])
            swap_col = int(wc[row, col])
            restored[row, col], restored[swap_row, swap_col] = (
                restored[swap_row, swap_col],
                restored[row, col],
            )
    return restored


def _generate_seed_from_key_image(key_image: np.ndarray) -> int:
    digest = hashlib.sha512(key_image.tobytes()).digest()
    key_bits = "".join(f"{byte:08b}" for byte in digest)
    groups = [key_bits[idx * 32 : (idx + 1) * 32] for idx in range(16)]

    xor_groups = []
    for start in range(0, 16, 4):
        value = 0
        for group in groups[start : start + 4]:
            value ^= int(group, 2)
        xor_groups.append(value)

    return sum(xor_groups) * (1 << 478)


def encrypt_image_with_debug(image: np.ndarray, leu: int) -> tuple[np.ndarray, EFIEDebugState]:
    ciphertext, debug_state = _encrypt_core(image=image, leu=leu, key_image=image, return_debug=True)
    if debug_state is None:
        raise RuntimeError("Debug state was not produced.")
    return ciphertext, debug_state


def load_rgb_image(path: str | Path) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to load images from disk.")
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    _validate_rgb_image(image, "image")
    if Image is None:
        raise ImportError("Pillow is required to save images to disk.")
    Image.fromarray(image, mode="RGB").save(path)


class Encrypter:
    """
    Thin compatibility wrapper around the EFIE function library.

    `key` and `img_path` may be numpy arrays or file paths. If `key` is omitted,
    encryption uses the plaintext image itself as the key image, matching the
    currently requested behavior.
    """

    def __init__(self, key: Any = None, img_path: Any = None):
        self.key = key
        self.img_path = img_path

    def encrypt(self, image: np.ndarray | None = None, leu: int = 1) -> np.ndarray:
        plaintext = self._coerce_image(image if image is not None else self.img_path, "image")
        if self.key is not None:
            key_image = self._coerce_image(self.key, "key")
            return _encrypt_core(plaintext, leu, key_image, return_debug=False)[0]
        return encrypt_image(plaintext, leu)

    def decrypt(
        self,
        ciphertext: np.ndarray | None = None,
        leu: int = 1,
        key_image: np.ndarray | None = None,
    ) -> np.ndarray:
        cipher = self._coerce_image(ciphertext if ciphertext is not None else self.img_path, "ciphertext")
        actual_key = key_image if key_image is not None else self.key
        if actual_key is None:
            raise ValueError("A key_image is required for decryption.")
        key = self._coerce_image(actual_key, "key_image")
        return decrypt_image(cipher, key, leu)

    @staticmethod
    def _coerce_image(value: Any, name: str) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            return load_rgb_image(value)
        raise TypeError(f"{name} must be a numpy array or an image path.")


def _run_self_test(sample_path: str | Path | None = None) -> None:
    if sample_path is None:
        sample_path = Path(__file__).with_name("self_test_sample.png")
    sample = load_rgb_image(sample_path)
    height, width, _ = sample.shape

    expected_grouped_shapes = {
        1: (height, width * 24),
        2: (height, width * 12),
        4: (height, width * 6),
        8: (height, width * 3),
    }

    for leu in ALLOWED_LEU:
        ciphertext, debug = encrypt_image_with_debug(sample, leu)
        restored = decrypt_image(ciphertext, sample, leu)
        if not np.array_equal(restored, sample):
            raise AssertionError(f"Roundtrip failed for leu={leu}.")
        if debug.i1_shape != (height, width * 3):
            raise AssertionError(f"Unexpected I1 shape for leu={leu}: {debug.i1_shape}")
        if debug.i2_shape != (height, width * 24) or debug.i3_shape != (height, width * 24):
            raise AssertionError(f"Unexpected I2/I3 shape for leu={leu}.")
        if debug.grouped_shape != expected_grouped_shapes[leu]:
            raise AssertionError(f"Unexpected grouped shape for leu={leu}: {debug.grouped_shape}")
        if debug.i4_shape != (height * width * 3, 8):
            raise AssertionError(f"Unexpected I4 shape for leu={leu}: {debug.i4_shape}")
        if debug.i6_shape != (height, width * 3):
            raise AssertionError(f"Unexpected I6 shape for leu={leu}: {debug.i6_shape}")

    try:
        encrypt_image(sample[:, :, 0], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-RGB image.")

    try:
        encrypt_image(sample.astype(np.int16), 1)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-uint8 image.")

    try:
        encrypt_image(sample, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid leu.")

    try:
        decrypt_image(sample, None, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing key_image.")

    print("EFIE self-test passed.")


if __name__ == "__main__":
    imagepth = "C:\\ImageEncryptionV2\\image\\img1.png"
    _run_self_test(imagepth)
