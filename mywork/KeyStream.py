from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from mpmath.ctx_mp_python import mpf
import random

from .SalomonCouplingCML import *

# 使用高精度浮点，避免在中间迭代阶段过早退化到双精度
mp.mp.dps = 80


def _parse_user_key_bits(user_key: str) -> np.ndarray:
    """将用户输入的 256 位二进制密钥转换为位数组。"""
    if not isinstance(user_key, str):
        raise TypeError("user_key must be a 256-character binary string")
    if len(user_key) != 256:
        raise ValueError("user_key must contain exactly 256 bits")
    if any(bit not in "01" for bit in user_key):
        raise ValueError("user_key must contain only '0' and '1'")

    return np.fromiter((int(bit) for bit in user_key), dtype=np.uint8, count=256)


def _derive_chaos_parameters(plaintext_bytes: bytes, user_key: str) -> dict[str, mpf]:
    """从明文图片的原始字节中派生高精度混沌参数。"""
    sha256_ctx = hashlib.sha256()
    sha256_ctx.update(plaintext_bytes)
    digest = sha256_ctx.digest()

    # 转成 256 位二进制并按固定分段提取参数
    image_hash_bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    user_key_bits = _parse_user_key_bits(user_key)
    H = np.bitwise_xor(image_hash_bits, user_key_bits)

    # 全局扰动因子（6 位）
    g_bits = H[250:256]
    g_int = sum(int(bit) << idx for idx, bit in enumerate(g_bits.tolist()))
    g = mp.mpf(g_int) / mp.power(2, 6)

    U: list[mpf] = []
    # 前 250 位分成 5 段，每段 50 位
    for j in range(5):
        start_idx = j * 50
        block_bits = H[start_idx : start_idx + 50]

        # 使用 Python 原生整数避免位信息损失
        block_int = sum(int(bit) << idx for idx, bit in enumerate(block_bits.tolist()))

        V_j = mp.mpf(block_int) / mp.power(2, 50)
        mix_val = mp.fmod(V_j + g, mp.mpf(1))
        U_j = mp.fabs(mp.sin(4 * mp.pi * mix_val))
        U.append(U_j)

    return {
        "mu": mp.mpf(10) * U[0],
        "v": mp.mpf(10) * U[1],
        "alpha": mp.mpf(10) * U[2],
        "beta": mp.mpf(10) * U[3],
        "x": U[4],
    }


def _validate_iteration_inputs(L: int, warmup: int) -> None:
    if not isinstance(L, (int, np.integer)) or L <= 0:
        raise ValueError("L must be a positive integer")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")



def _iterate_keystream_mp(x0: mpf, mu: mpf, v: mpf, L: int, warmup: int = 50) -> list[mpf]:
    """按给定公式进行高精度迭代，并丢弃前 warmup 次结果。"""
    _validate_iteration_inputs(L, warmup)

    total_steps = int(L) + warmup
    seq: list[mpf] = [mp.mpf(0)] * int(L)
    x = mp.mpf(x0)
    factor = mp.mpf(5) + mp.mpf(3) * mp.mpf(mu)
    v = mp.mpf(v)
    one = mp.mpf(1)

    for step_idx in range(total_steps):
        x = mp.fabs(
            mp.sin(
                factor
                * (
                    one
                    - (
                        v
                        * x
                        * mp.sin(mp.mpf(15) * mp.pi * x * (one - x))
                    )
                )
            )
        )
        if step_idx >= warmup:
            seq[step_idx - warmup] = x

    if len(seq) != L:
        raise RuntimeError(f"Unexpected keystream length: expected {L}, got {len(seq)}")

    return seq



def _iterate_keystream_float(x0: float, mu: float, v: float, L: int, warmup: int = 50) -> np.ndarray:
    """按给定公式进行轻量级 float 迭代，并丢弃前 warmup 次结果。"""
    _validate_iteration_inputs(L, warmup)

    total_steps = int(L) + warmup
    seq = np.empty(int(L), dtype=np.float64)
    x = float(x0)
    factor = 5.0 + 3.0 * float(mu)
    v = float(v)

    for step_idx in range(total_steps):
        x = abs(
            math.sin(
                factor
                * (
                    1.0
                    - (
                        v
                        * x
                        * math.sin(15.0 * math.pi * x * (1.0 - x))
                    )
                )
            )
        )
        if step_idx >= warmup:
            seq[step_idx - warmup] = x

    if seq.size != L:
        raise RuntimeError(f"Unexpected keystream length: expected {L}, got {seq.size}")

    return seq



def _finalize_output(params: dict[str, mpf], sequence: list[mpf] | np.ndarray) -> dict[str, float | np.ndarray]:
    """仅在最终输出阶段降为常规浮点类型。"""
    scalar_names = ("mu", "v", "alpha", "beta")
    scalar_values = np.fromiter(
        (float(params[name]) for name in scalar_names),
        dtype=np.float64,
        count=len(scalar_names),
    )

    if isinstance(sequence, np.ndarray):
        sequence_array = np.asarray(sequence, dtype=np.float64)
    else:
        sequence_array = np.fromiter(
            (float(value) for value in sequence),
            dtype=np.float64,
            count=len(sequence),
        )

    return {
        scalar_names[idx]: scalar_values[idx] for idx in range(len(scalar_names))
    } | {"X": sequence_array}



def keystream_generation(
    L: int,
    plaintext_image_path: str | Path,
    user_key: str,
    *,
    use_high_precision: bool = False,
) -> dict[str, float | np.ndarray]:
    """根据明文图片和长度 L 生成混沌参数及随机数序列。

    返回字段：
        mu, v, alpha, beta, X
    其中 X 为长度为 L 的随机数序列，x 仅作为内部初值使用。

    参数：
        use_high_precision=False 时使用轻量级 float 迭代；
        use_high_precision=True 时使用高精度 mp 迭代。
    """
    if not isinstance(L, (int, np.integer)) or L <= 0:
        raise ValueError("L must be a positive integer")

    image_path = Path(plaintext_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Plaintext image not found: {image_path}")

    plaintext_bytes = image_path.read_bytes()
    if not plaintext_bytes:
        raise ValueError(f"Plaintext image is empty: {image_path}")

    params = _derive_chaos_parameters(plaintext_bytes, user_key)
    if use_high_precision:
        sequence = _iterate_keystream_mp(
            x0=params["x"],
            mu=params["mu"],
            v=params["v"],
            L=int(L),
            warmup=50,
        )
    else:
        sequence = _iterate_keystream_float(
            x0=float(params["x"]),
            mu=float(params["mu"]),
            v=float(params["v"]),
            L=int(L),
            warmup=50,
        )
    return _finalize_output(params, sequence)


# --- 实验验证 ---
if __name__ == "__main__":
    sample_path = Path(r"C:\ImageEncryption\images\img2.png")
    if sample_path.is_file():
        import time
        seed = 2026
        random.seed(seed)
        binary_key_str = format(random.getrandbits(256), "0256b")
        hex_key = f"{int(binary_key_str, 2):064x}"
        print(hex_key)

        user_key = binary_key_str
        st = time.time()
        result = keystream_generation(512 * 512, sample_path, user_key)
        et = time.time()
        print(f"Time taken: {et - st:.6f} seconds")
        print("====== 混沌参数与随机数序列输出 ======")
        for key in ("mu", "v", "alpha", "beta"):
            print(f"{key:<6}: {result[key]:.30f}")
        print(f"X len : {len(result['X'])}")
        print(f"X head: {np.array2string(result['X'][:10], precision=8, separator=', ')}")
        
    else:
        print(f"Sample image not found: {sample_path}")
