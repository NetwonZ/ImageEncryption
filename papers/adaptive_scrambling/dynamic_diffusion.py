"""论文Algorithm 3：动态S-box XOR与spiral cumulative diffusion。"""

from __future__ import annotations

import numpy as np


def generate_dynamic_sbox(x: np.ndarray) -> tuple[np.ndarray, int]:
    """依据论文公式(3)生成由seed确定的动态S-box。"""
    value = int(np.floor(abs(float(np.sum(x, dtype=np.float64))) * 1e6))
    seed = value % (2 ** 32)
    rng = np.random.default_rng(seed)
    sbox = rng.permutation(np.arange(256, dtype=np.uint8))
    return sbox, seed


def spiral_path(height: int, width: int) -> np.ndarray:
    """返回论文Fig.9所示的顺时针、外向内spiral path索引。"""
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正")
    top, bottom, left, right = 0, height - 1, 0, width - 1
    path: list[tuple[int, int]] = []
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            path.append((top, col))
        top += 1
        for row in range(top, bottom + 1):
            path.append((row, right))
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                path.append((bottom, col))
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                path.append((row, left))
            left += 1
    return np.asarray([r * width + c for r, c in path], dtype=np.int64)


def local_xor(scrambled: np.ndarray, x: np.ndarray) -> np.ndarray:
    """执行论文Algorithm 3第3步：S[k] XOR sBox[index]。"""
    image = np.asarray(scrambled, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("输入必须是二维灰度图像")
    flat = image.reshape(-1)
    if len(x) < len(flat):
        raise ValueError("混沌序列长度不足")
    sbox, _ = generate_dynamic_sbox(x)
    indices = (np.floor(np.abs(x[: len(flat)]) * 1e6).astype(np.uint64) % 256).astype(np.int64)
    return np.bitwise_xor(flat, sbox[indices]).reshape(image.shape).astype(np.uint8)


def inverse_local_xor(diffused: np.ndarray, x: np.ndarray) -> np.ndarray:
    """XOR自反，执行local XOR的逆过程。"""
    return local_xor(diffused, x)


def spiral_diffusion(local: np.ndarray, z: np.ndarray) -> np.ndarray:
    """执行论文公式(5)：Ck=mod(Ck+Ck-1+dk,256)。"""
    image = np.asarray(local, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("输入必须是二维灰度图像")
    n = image.size
    if len(z) < n:
        raise ValueError("混沌序列长度不足")
    path = spiral_path(*image.shape)
    c = image.reshape(-1).astype(np.uint16).copy()
    d = (np.floor(np.abs(z[:n]) * 1e6).astype(np.uint64) % 256).astype(np.uint16)
    for k in range(1, n):
        c[path[k]] = (c[path[k]] + c[path[k - 1]] + d[k]) % 256
    return c.reshape(image.shape).astype(np.uint8)


def inverse_spiral_diffusion(cipher: np.ndarray, z: np.ndarray) -> np.ndarray:
    """按k=N...2逆解论文公式(5)。"""
    image = np.asarray(cipher, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("输入必须是二维灰度图像")
    n = image.size
    if len(z) < n:
        raise ValueError("混沌序列长度不足")
    path = spiral_path(*image.shape)
    c = image.reshape(-1).astype(np.int64).copy()
    d = (np.floor(np.abs(z[:n]) * 1e6).astype(np.uint64) % 256).astype(np.int64)
    for k in range(n - 1, 0, -1):
        c[path[k]] = (c[path[k]] - c[path[k - 1]] - d[k]) % 256
    return c.reshape(image.shape).astype(np.uint8)


def diffuse_encrypt(scrambled: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """执行完整动态交叉扩散，返回(ciphertext, local_xor_result, path, seed)。"""
    sbox, seed = generate_dynamic_sbox(x)
    local = local_xor(scrambled, x)
    cipher = spiral_diffusion(local, z)
    return cipher, local, sbox, seed


def diffuse_decrypt(cipher: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """完整逆扩散：先逆spiral，再逆XOR。"""
    local = inverse_spiral_diffusion(cipher, z)
    return inverse_local_xor(local, x)
