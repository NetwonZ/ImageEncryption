"""统一对外接口：从四个文献复现的子模块里拉取性能指标。

- 每个 ``PAPERS[name]`` 都是 ``fn(image_paths) -> list[float]``；
- 脚本里 import：内部调用对应文献目录的 ``performance_analysis.cipher_histogram_variance``；
- compare.py 直接遍历 ``PAPERS`` 打印对比表格，不再管理导入细节。

注意：四个文献目录各自都有名为 ``src`` 的包与名为 ``encryption``/``performance_analysis``
的脚本（命名空间包冲突），所以这里用
``importlib.util.spec_from_file_location`` 按绝对路径加载脚本模块，并在切换文献时
清掉 ``sys.modules`` 里缓存的 ``src``/``encryption`` 条目，避免跨目录同名模块相互覆盖。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(r"C:\ImageEncryption")
PAPERS_DIR = ROOT / "papers"

_PAPER_SCRIPTS = {
    "Yu 2026 自适应置乱（memristive 混沌神经元）": PAPERS_DIR
    / "adaptive_scrambling",
    "Hao 2025 混合混沌分层彩色加密": PAPERS_DIR
    / "ColorImageEncryptionAlgorithmBasedonHybridChaosandLayered_2025",
    "Wu 陈氏超混沌 OTWFM": PAPERS_DIR
    / "ImageCryptosystemBasedonHyperChaosandOptimizedTyphoon",
    "Li & Tao 2026 LSTNet 多图超混沌": PAPERS_DIR
    / "ImprovedLSTNet-DrivenHyperchaoticSequenceOptimizationAndItsApplicationinMultiImageEncrypt_2026",
}

_loaded: dict[str, callable] = {}

# 各文献实现共用的顶层模块名；切换文献前必须清掉，否则 import 会撞到上一份的缓存
_COLLIDING_NAMES = ("src", "encryption", "performance_analysis", "cipher", "pipeline")


def _purge_colliding_modules() -> None:
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in _COLLIDING_NAMES:
            del sys.modules[name]


def _load_histogram_variance(paper_dir: Path):
    """按文件路径加载 paper 的 ``performance_analysis.py``，返回其 HV 函数。"""
    cached = _loaded.get(str(paper_dir))
    if cached is not None:
        return cached
    script = paper_dir / "performance_analysis.py"
    _purge_colliding_modules()
    sys.path.insert(0, str(paper_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"performance_analysis_{abs(hash(str(paper_dir)))}", script
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module.cipher_histogram_variance
        _loaded[str(paper_dir)] = fn
    finally:
        sys.path.remove(str(paper_dir))
        _purge_colliding_modules()
    return fn


def _bind(paper_dir: Path):
    def _fn(image_paths: list[str]) -> list[float]:
        return _load_histogram_variance(paper_dir)(image_paths)

    return _fn


# compare.py 只消费这个 dict
PAPERS = {name: _bind(path) for name, path in _PAPER_SCRIPTS.items()}


def print_ascii_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print +---+ 分隔的 ASCII table；rows 顺序即打印顺序。

    通过南华 echarts-style ``ljust/rjust`` 保持列对齐，宽度由内容决定。
    """
    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError("headers/rows 形状不一致")
    text_rows = [tuple(str(col) for col in row) for row in rows]
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in text_rows))
        for i in range(len(headers))
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(row: tuple[str, ...], *, header: bool = False):
        cells = []
        for i, col in enumerate(row):
            cells.append(col.center(widths[i]) if header else col.ljust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    print(sep)
    print(fmt(headers, header=True))
    print(sep.replace("-", "="))
    for row in text_rows:
        print(fmt(row))
    print(sep)
