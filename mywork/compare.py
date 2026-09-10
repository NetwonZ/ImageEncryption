"""四篇文献密文直方图方差横向对比。

约定：输入 PLAIN_IMAGES（list[str]），每个性能函数返回 list[float]，
打印表格列出论文名字、对应明文图像、HV；每张明文都给出 MEAN 供横评。
"""

from __future__ import annotations

from pathlib import Path

from analysis_api import PAPERS, print_ascii_table

PLAIN_IMAGES = [
    r"C:\ImageEncryption\images\img8.png",
    r"C:\ImageEncryption\images\img3.png",
    r"C:\ImageEncryption\images\img5.png",
]


def main(plain_paths: list[str] | None = None) -> None:
    paths = plain_paths or PLAIN_IMAGES
    rows: list[tuple[str, str, str]] = []
    for paper_name, hv_fn in PAPERS.items():
        hvs = hv_fn(paths)
        for path, hv in zip(paths, hvs):
            rows.append((paper_name, Path(path).name, f"{hv:.2f}"))
        mean_hv = sum(hvs) / len(hvs)
        rows.append((paper_name, "**MEAN**", f"{mean_hv:.2f}"))

    print_ascii_table(("论文", "图像", "HV(密文)"), rows)


if __name__ == "__main__":
    main()
