"""忆阻混沌神经元：论文式(1)与Euler离散实现。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NeuronParameters:
    """论文给出的忆阻混沌神经元参数。"""

    a: float = 0.7
    b: float = 0.8
    c: float = 0.1
    alpha: float = 0.25
    beta: float = 0.01
    delta: float = 0.1
    A: float = 0.35
    f: float = 0.7
    x0: float = 0.2
    y0: float = 0.0
    z0: float = 0.01


def derivatives(t: float, x: float, y: float, z: float,
                params: NeuronParameters) -> tuple[float, float, float]:
    """返回论文公式(1)的三个右端项。外部刺激 us=A cos(f*t)。"""
    us = params.A * np.cos(params.f * t)
    dx = x - params.a * x - (x ** 3) / 3.0 - y + us
    dy = params.c * x - params.b * params.c * np.sin(params.beta * z) * y
    dz = params.delta * y - params.alpha * z
    return float(dx), float(dy), float(dz)


def generate_chaotic_sequences(
    n: int,
    *,
    dt: float = 0.01,
    pre_iterations: int = 1000,
    params: NeuronParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按论文Algorithm 1生成长度为n的x、y、z序列。

    论文给出Euler更新形式，但未在正文公开dt和预迭代次数；二者保留为
    显式参数，默认值用于可重复运行。序列从预迭代后的状态开始记录。
    """
    if n <= 0:
        raise ValueError("n必须为正整数")
    if dt <= 0:
        raise ValueError("dt必须为正数")
    if pre_iterations < 0:
        raise ValueError("pre_iterations不能为负数")
    params = params or NeuronParameters()
    total_steps = pre_iterations + n
    x_seq = np.empty(n, dtype=np.float64)
    y_seq = np.empty(n, dtype=np.float64)
    z_seq = np.empty(n, dtype=np.float64)
    x, y, z = params.x0, params.y0, params.z0

    for k in range(total_steps):
        dx, dy, dz = derivatives(k * dt, x, y, z, params)
        x += dt * dx
        y += dt * dy
        z += dt * dz
        if k >= pre_iterations:
            idx = k - pre_iterations
            x_seq[idx], y_seq[idx], z_seq[idx] = x, y, z
    return x_seq, y_seq, z_seq


def seed_from_x(x: np.ndarray) -> int:
    """论文公式(3)：seed=mod(floor(|sum(x)|*10^6),2^32)。"""
    value = int(np.floor(abs(float(np.sum(x, dtype=np.float64))) * 1e6))
    return value % (2 ** 32)


def seed_from_chaos(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> int:
    """由三条混沌序列生成置乱阶段的确定性随机种子。"""
    value = int(np.floor(abs(float(np.sum(x + y + z, dtype=np.float64))) * 1e6))
    return value % (2 ** 32)
