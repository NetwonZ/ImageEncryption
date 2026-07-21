"""完整的论文图像加密/解密流程。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .adaptive_scrambling import ScrambleKey, key_summary, scramble_rounds, unscramble_rounds
from .chaotic_neuron import NeuronParameters, generate_chaotic_sequences, seed_from_chaos
from .dynamic_diffusion import diffuse_decrypt, diffuse_encrypt


@dataclass
class EncryptionResult:
    ciphertext: np.ndarray
    scrambled: np.ndarray
    local_diffusion: np.ndarray
    decrypted: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    scramble_keys: list[ScrambleKey]
    sbox: np.ndarray
    diffusion_seed: int


class ImageCryptosystem:
    """论文流程的可逆实现。

    dt、预迭代次数及分块尺寸通过构造参数暴露，便于对照论文或复现实验。
    """

    def __init__(self, *, dt: float = 0.01, pre_iterations: int = 1000,
                 params: NeuronParameters | None = None) -> None:
        self.dt = dt
        self.pre_iterations = pre_iterations
        self.params = params or NeuronParameters()

    def encrypt(self, image: np.ndarray) -> EncryptionResult:
        plain = np.asarray(image, dtype=np.uint8)
        if plain.ndim != 2 or plain.shape[0] != plain.shape[1]:
            raise ValueError("论文实现要求输入为正方形二维灰度图像")
        n = plain.size
        x, y, z = generate_chaotic_sequences(
            n, dt=self.dt, pre_iterations=self.pre_iterations, params=self.params
        )
        base_seed = seed_from_chaos(x, y, z)
        scramble_seeds = (
            base_seed,
            (base_seed ^ 0x9E3779B9) % (2 ** 32),
        )
        scrambled, keys = scramble_rounds(plain, seeds=scramble_seeds)
        ciphertext, local, sbox, diffusion_seed = diffuse_encrypt(scrambled, x, z)
        decrypted = self.decrypt(ciphertext, x=x, y=y, z=z, scramble_keys=keys)
        return EncryptionResult(ciphertext, scrambled, local, decrypted, x, y, z,
                                keys, sbox, diffusion_seed)

    def decrypt(self, ciphertext: np.ndarray, *, x: np.ndarray, y: np.ndarray,
                z: np.ndarray, scramble_keys: list[ScrambleKey]) -> np.ndarray:
        del y  # y参与混沌生成，但扩散公式使用论文指定的x、z
        unscrambled = diffuse_decrypt(np.asarray(ciphertext, dtype=np.uint8), x, z)
        return unscramble_rounds(unscrambled, scramble_keys)


def debug_summary(result: EncryptionResult) -> dict:
    """输出/记录便于调试的中间变量摘要。"""
    return {
        "shape": tuple(result.ciphertext.shape),
        "sequence_length": len(result.x),
        "x_head": np.round(result.x[:5], 8).tolist(),
        "y_head": np.round(result.y[:5], 8).tolist(),
        "z_head": np.round(result.z[:5], 8).tolist(),
        "chaos_ranges": {
            "x": [float(np.min(result.x)), float(np.max(result.x))],
            "y": [float(np.min(result.y)), float(np.max(result.y))],
            "z": [float(np.min(result.z)), float(np.max(result.z))],
        },
        "diffusion_seed": result.diffusion_seed,
        "sbox_head": result.sbox[:16].tolist(),
        "scrambling": key_summary(result.scramble_keys),
    }
