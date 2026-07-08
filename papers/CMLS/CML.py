from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def cml_step(x, r=4.0, eps=0.2):
    """
    一维Logistic耦合映射晶格
    周期边界条件
    """
    fx = r * x * (1.0 - x)

    left = np.roll(fx, 1)
    right = np.roll(fx, -1)

    x_next = (1.0 - eps) * fx + 0.5 * eps * (left + right)

    return x_next


def cml_jacobian(x, r=4.0, eps=0.2):
    """
    返回CML当前状态的Jacobian矩阵
    """
    N = len(x)

    fp = r * (1.0 - 2.0 * x)

    J = np.zeros((N, N))

    for i in range(N):
        J[i, i] = (1.0 - eps) * fp[i]

        left = (i - 1) % N
        right = (i + 1) % N

        J[i, left] = 0.5 * eps * fp[left]
        J[i, right] = 0.5 * eps * fp[right]

    return J


def lyapunov_spectrum(
    N=100,
    r=4.0,
    eps=0.2,
    transient=5000,
    steps=50000,
    seed=None,
    show_progress=True,
):
    rng = np.random.default_rng(seed)
    x = rng.random(N)

    Q = np.eye(N)
    sums = np.zeros(N)

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            transient_task = progress.add_task("Transient", total=transient)

            for _ in range(transient):
                x = cml_step(x, r, eps)
                progress.update(transient_task, advance=1)

            le_task = progress.add_task("Lyapunov Spectrum", total=steps)

            for _ in range(steps):
                J = cml_jacobian(x, r, eps)
                Z = J @ Q
                Q, R = np.linalg.qr(Z)
                sums += np.log(np.abs(np.diag(R)))
                x = cml_step(x, r, eps)
                progress.update(le_task, advance=1)
    else:
        for _ in range(transient):
            x = cml_step(x, r, eps)

        for _ in range(steps):
            J = cml_jacobian(x, r, eps)
            Z = J @ Q
            Q, R = np.linalg.qr(Z)
            sums += np.log(np.abs(np.diag(R)))
            x = cml_step(x, r, eps)

    le = sums / steps
    le = np.sort(le)[::-1]

    return le


def compute_ked_keb(spectrum):
    """
    根据李雅普诺夫谱计算：

    KED : Kolmogorov-Sinai entropy density
    KEB : Kolmogorov-Sinai entropy breadth
    """
    le = np.asarray(spectrum, dtype=np.float64)

    N = le.size
    if N == 0:
        raise ValueError("Spectrum is empty.")

    positive = le[le > 0]

    if positive.size == 0:
        ked = 0.0
        keb = 0.0
    else:
        ked = positive.sum() / positive.size
        keb = positive.size / N

    return ked, keb


def scan_parameter_plane(
    r_values,
    eps_values,
    N=100,
    transient=5000,
    steps=50000,
    seed=None,
    save_path=None,
):
    """
    扫描(r, eps)参数平面，计算并保存KED/KEB。

    Parameters
    ----------
    r_values : array-like
        r参数采样点。
    eps_values : array-like
        eps参数采样点。
    save_path : str or Path, optional
        本地保存路径，默认保存在当前文件同级output目录下。

    Returns
    -------
    result : dict
        包含保存路径、参数轴和KED/KEB矩阵。
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    eps_values = np.asarray(eps_values, dtype=np.float64)

    ked_map = np.empty((eps_values.size, r_values.size), dtype=np.float64)
    keb_map = np.empty_like(ked_map)

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "results" / "cml_parameter_scan.npz"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(seed)
    total = int(r_values.size * eps_values.size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        scan_task = progress.add_task("Parameter Scan", total=total)

        for i, eps in enumerate(eps_values):
            for j, r in enumerate(r_values):
                point_seed = int(base_rng.integers(0, np.iinfo(np.int64).max))
                spectrum = lyapunov_spectrum(
                    N=N,
                    r=float(r),
                    eps=float(eps),
                    transient=transient,
                    steps=steps,
                    seed=point_seed,
                    show_progress=False,
                )
                ked, keb = compute_ked_keb(spectrum)
                ked_map[i, j] = ked
                keb_map[i, j] = keb
                progress.update(scan_task, advance=1)

    np.savez_compressed(
        save_path,
        r_values=r_values,
        eps_values=eps_values,
        ked_map=ked_map,
        keb_map=keb_map,
        N=N,
        transient=transient,
        steps=steps,
        seed=-1 if seed is None else int(seed),
    )

    return {
        "save_path": str(save_path),
        "r_values": r_values,
        "eps_values": eps_values,
        "ked_map": ked_map,
        "keb_map": keb_map,
    }


def plot_parameter_wireframe(
    data_path,
    quantity="ked",
    output_path=None,
    stride=1,
    figsize=(10, 7),
):
    """
    从本地保存的参数扫描结果中读取数据并绘制线框图。

    Parameters
    ----------
    data_path : str or Path
        由scan_parameter_plane生成的npz文件。
    quantity : {"ked", "keb"}
        选择绘制的量。
    output_path : str or Path, optional
        图片输出路径，默认与数据文件同名。
    stride : int
        线框抽样步长。

    Returns
    -------
    fig, ax, output_path
    """
    data_path = Path(data_path)
    data = np.load(data_path)

    r_values = data["r_values"]
    eps_values = data["eps_values"]
    z_map = data[f"{quantity}_map"]

    r_grid, eps_grid = np.meshgrid(r_values, eps_values)

    if output_path is None:
        output_path = data_path.with_name(f"{data_path.stem}_{quantity}_wireframe.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(
        r_grid,
        eps_grid,
        z_map,
        rstride=stride,
        cstride=stride,
        linewidth=0.7,
        color="tab:blue",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("eps")
    ax.set_zlabel(quantity.upper())
    ax.set_title(f"{quantity.upper()} wireframe")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax, str(output_path)


if __name__ == "__main__":
    r_values = np.linspace(3.0, 4.0, 20)
    eps_values = np.linspace(0.0, 1.0, 20)

    result = scan_parameter_plane(
        r_values=r_values,
        eps_values=eps_values,
        N=100,
        transient=50,
        steps=300,
        seed=2026,
    )
    print("Saved scan to:", result["save_path"])

    _, _, figure_path = plot_parameter_wireframe(result["save_path"], quantity="ked")
    print("Saved figure to:", figure_path)
