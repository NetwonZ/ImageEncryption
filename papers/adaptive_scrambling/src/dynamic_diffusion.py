"""论文Algorithm 3：动态S-box XOR与spiral cumulative diffusion。"""

from __future__ import annotations

import numpy as np


PAPER_MODE = "paper"
HARDENED_MODE = "hardened"
_VALID_MODES = {PAPER_MODE, HARDENED_MODE}


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


def nonlinear_bidirectional_diffusion(local: np.ndarray, x: np.ndarray,
                                      z: np.ndarray) -> np.ndarray:
    """增强模式：可逆的双向非线性螺旋交叉扩散。

    原论文公式(5)为单向线性累加，无法让单点差异产生接近理想值的 UACI。
    本模式保留动态 S-box 与螺旋路径，并增加两个互逆的反馈阶段：

        F_k = (A_k + SBox[F_{k-1}] + d_k) mod 256
        C_k = (F_k + SBox[C_{k+1}] + d_k) mod 256

    第一式沿螺旋正向计算，第二式沿反向计算。S-box 是置换，因而反馈项
    非线性且不会像 ``p + SBox[p]`` 一样发生状态合并；两个方向共同保证
    任意位置的单点修改能影响整条螺旋路径。
    """
    image = np.asarray(local, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("输入必须是二维灰度图像")
    n = image.size
    if len(x) < n or len(z) < n:
        raise ValueError("混沌序列长度不足")

    sbox, _ = generate_dynamic_sbox(x)
    path = spiral_path(*image.shape)
    d = (np.floor(np.abs(z[:n]) * 1e6).astype(np.uint64) % 256).astype(np.int64)

    forward = image.reshape(-1).astype(np.int64).copy()
    for k in range(1, n):
        previous = forward[path[k - 1]]
        forward[path[k]] = (forward[path[k]] + int(sbox[previous]) + d[k]) % 256

    cipher = forward.copy()
    for k in range(n - 2, -1, -1):
        following_cipher = cipher[path[k + 1]]
        cipher[path[k]] = (cipher[path[k]] + int(sbox[following_cipher]) + d[k]) % 256
    return cipher.reshape(image.shape).astype(np.uint8)


def inverse_nonlinear_bidirectional_diffusion(cipher: np.ndarray, x: np.ndarray,
                                              z: np.ndarray) -> np.ndarray:
    """逆解 :func:`nonlinear_bidirectional_diffusion`。"""
    image = np.asarray(cipher, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("输入必须是二维灰度图像")
    n = image.size
    if len(x) < n or len(z) < n:
        raise ValueError("混沌序列长度不足")

    sbox, _ = generate_dynamic_sbox(x)
    path = spiral_path(*image.shape)
    d = (np.floor(np.abs(z[:n]) * 1e6).astype(np.uint64) % 256).astype(np.int64)
    cipher_flat = image.reshape(-1).astype(np.int64)

    # 逆第二阶段。右侧使用原始C[k+1]，因此不能在同一数组上覆盖它。
    forward = cipher_flat.copy()
    for k in range(n - 2, -1, -1):
        forward[path[k]] = (
            cipher_flat[path[k]] - int(sbox[cipher_flat[path[k + 1]]]) - d[k]
        ) % 256

    # 逆第一阶段。此时forward已完整恢复，使用F[k-1]即可逐点恢复A[k]。
    local = forward.copy()
    for k in range(1, n):
        local[path[k]] = (
            forward[path[k]] - int(sbox[forward[path[k - 1]]]) - d[k]
        ) % 256
    return local.reshape(image.shape).astype(np.uint8)


def diffuse_encrypt(
    scrambled: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    mode: str = HARDENED_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """执行动态交叉扩散。

    ``mode='paper'`` 严格执行论文公式(5)；``mode='hardened'`` 使用
    双向非线性反馈以提供可验证的抗差分能力。
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"未知扩散模式: {mode}，可选值为{sorted(_VALID_MODES)}")
    sbox, seed = generate_dynamic_sbox(x)
    local = local_xor(scrambled, x)
    cipher = (
        spiral_diffusion(local, z)
        if mode == PAPER_MODE
        else nonlinear_bidirectional_diffusion(local, x, z)
    )
    return cipher, local, sbox, seed


def diffuse_decrypt(
    cipher: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    mode: str = HARDENED_MODE,
) -> np.ndarray:
    """完整逆扩散：先逆螺旋反馈，再逆XOR。"""
    if mode not in _VALID_MODES:
        raise ValueError(f"未知扩散模式: {mode}，可选值为{sorted(_VALID_MODES)}")
    local = (
        inverse_spiral_diffusion(cipher, z)
        if mode == PAPER_MODE
        else inverse_nonlinear_bidirectional_diffusion(cipher, x, z)
    )
    return inverse_local_xor(local, x)
