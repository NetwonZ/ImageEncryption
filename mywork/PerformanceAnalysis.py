"""Reusable performance and security analysis for image-encryption callables."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_image

ImageInput = Union[str, Path, np.ndarray, pil_image.Image]
CryptFunction = Callable[..., Any]

_DIRECTIONS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diagonal": (1, 1),
}


def _print_ascii_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    """Print a compact ASCII table with a left-aligned first column."""
    if not headers:
        return
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every table row must have the same number of values as headers")

    text_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(values: tuple[str, ...], *, header: bool = False) -> str:
        cells = []
        for index, value in enumerate(values):
            if header:
                cells.append(value.center(widths[index]))
            elif index == 0:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "| " + " | ".join(cells) + " |"

    print(separator)
    print(format_row(headers, header=True))
    print(separator)
    for row in text_rows:
        print(format_row(row))
    print(separator)


def _load_image_array(image: ImageInput) -> np.ndarray:
    """Load a path or ndarray into a grayscale or RGB ndarray."""
    if isinstance(image, (str, Path)):
        with pil_image.open(image) as img:
            if len(img.getbands()) == 1:
                return np.asarray(img.convert("L"))
            return np.asarray(img.convert("RGB"))

    if isinstance(image, pil_image.Image):
        if len(image.getbands()) == 1:
            return np.asarray(image.convert("L"))
        return np.asarray(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                return arr[:, :, 0]
            return arr[:, :, :3]
        raise ValueError("ndarray image must be 2D grayscale or 3D with 1, 3, or 4 channels")

    raise TypeError("image must be a file path, PIL image, or numpy ndarray")


def _to_uint8_pixels(arr: np.ndarray) -> np.ndarray:
    """Convert common image array dtypes to uint8 pixels for 0-255 histograms."""
    if arr.dtype == np.uint8:
        return arr

    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255

    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            finite = finite * 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)

    return np.clip(arr, 0, 255).astype(np.uint8)


def _pixel_pairs(channel: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent pixel pairs for one direction from a 2D image channel."""
    if direction == "horizontal":
        return channel[:, :-1].ravel(), channel[:, 1:].ravel()
    if direction == "vertical":
        return channel[:-1, :].ravel(), channel[1:, :].ravel()
    if direction == "diagonal":
        return channel[:-1, :-1].ravel(), channel[1:, 1:].ravel()
    raise ValueError(f"unknown direction: {direction}")


def _sample_pairs(
    x: np.ndarray,
    y: np.ndarray,
    sample_size: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample corresponding x/y pixel pairs without replacement."""
    if sample_size is None or sample_size >= x.size:
        return x, y
    indices = rng.choice(x.size, size=sample_size, replace=False)
    return x[indices], y[indices]


def _correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute corr(x, y) using the paper's E/D definition."""
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = np.mean((x - x_mean) ** 2)
    dy = np.mean((y - y_mean) ** 2)
    if dx == 0.0 or dy == 0.0:
        return 0.0
    covariance = np.mean((x - x_mean) * (y - y_mean))
    return float(covariance / np.sqrt(dx * dy))


def _information_entropy(pixels: np.ndarray) -> float:
    """Calculate Shannon information entropy in bits for uint8 pixels."""
    flat_pixels = pixels.ravel()
    if flat_pixels.size == 0:
        raise ValueError("image must contain at least one pixel")

    counts = np.bincount(flat_pixels, minlength=256)
    probabilities = counts[counts > 0] / flat_pixels.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return 0.0 if np.isclose(entropy, 0.0) else entropy


def _chi_square_value(pixels: np.ndarray) -> float:
    """Calculate the chi-square value for a uint8 image channel."""
    flat_pixels = pixels.ravel()
    if flat_pixels.size == 0:
        raise ValueError("image must contain at least one pixel")

    observed = np.bincount(flat_pixels, minlength=256).astype(np.float64)
    expected = flat_pixels.size / 256.0
    return float(np.sum((observed - expected) ** 2 / expected))


def calculate_information_entropy(image: ImageInput) -> dict[str, float]:
    """
    Calculate image information entropy.

    For encrypted 8-bit images, entropy closer to 8 indicates a more uniform
    pixel distribution. Grayscale images return ``{"Gray": entropy}``; RGB
    images return per-channel entropy plus ``average`` and ``overall`` values.

    Parameters
    ----------
    image:
        Encrypted image path or numpy ndarray.

    Returns
    -------
    dict[str, float]
        Information entropy values in bits.
    """
    arr = _to_uint8_pixels(_load_image_array(image))

    if arr.ndim == 2:
        return {"Gray": _information_entropy(arr)}

    entropies = {
        "R": _information_entropy(arr[:, :, 0]),
        "G": _information_entropy(arr[:, :, 1]),
        "B": _information_entropy(arr[:, :, 2]),
    }
    entropies["average"] = float(np.mean([entropies["R"], entropies["G"], entropies["B"]]))
    entropies["overall"] = _information_entropy(arr)
    return entropies


def calculate_chi_square_test(
    image: ImageInput,
    *,
    alpha: float = 0.05,
    critical_value: float | None = None,
    print_result: bool = True,
) -> dict[str, dict[str, float | bool]]:
    """
    Calculate the chi-square histogram uniformity test for an encrypted image.

    The test uses 256 gray levels. For each channel, the observed frequency is
    the channel histogram, and the expected frequency is ``pixel_count / 256``.
    At ``alpha=0.05`` with 255 degrees of freedom, the commonly used critical
    value is 293.2478. A smaller chi-square value indicates a more uniform
    histogram.

    Parameters
    ----------
    image:
        Encrypted image path or numpy ndarray.
    alpha:
        Significance level. The built-in critical value is provided for 0.05.
    critical_value:
        Optional custom threshold. If omitted, ``alpha=0.05`` uses 293.2478.
    print_result:
        Whether to print a table with check marks.

    Returns
    -------
    dict[str, dict[str, float | bool]]
        Per-channel chi-square values and pass/fail flags, plus ``average``.
    """
    if critical_value is None:
        if not np.isclose(alpha, 0.05):
            raise ValueError("critical_value must be provided when alpha is not 0.05")
        critical_value = 293.2478

    arr = _to_uint8_pixels(_load_image_array(image))

    if arr.ndim == 2:
        channel_items = [("Gray", arr)]
    else:
        channel_items = [
            ("R", arr[:, :, 0]),
            ("G", arr[:, :, 1]),
            ("B", arr[:, :, 2]),
        ]

    results: dict[str, dict[str, float | bool]] = {}
    values = []
    for channel_name, channel in channel_items:
        chi_square = _chi_square_value(channel)
        values.append(chi_square)
        results[channel_name] = {
            "chi_square": chi_square,
            "passed": chi_square <= critical_value,
        }

    average = float(np.mean(values))
    results["average"] = {
        "chi_square": average,
        "passed": average <= critical_value,
    }

    if print_result:
        print("\nChi-square histogram uniformity")
        print("--------------------------------")
        _print_ascii_table(
            ("Channel", "Chi-square", "Critical Value", "Alpha", "Result"),
            [
                (
                    "Average" if channel_name == "average" else channel_name,
                    f"{result['chi_square']:.4f}",
                    f"{critical_value:.4f}",
                    f"{alpha:.4f}",
                    "Pass" if result["passed"] else "Fail",
                )
                for channel_name, result in results.items()
            ],
        )

    return results


def plot_pixel_histogram(
    image: ImageInput,
    *,
    title: str = "Pixel Histogram",
    show: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """
    Plot the pixel histogram of a grayscale or color image.

    Parameters
    ----------
    image:
        Image path or numpy ndarray. A 2D array is treated as grayscale; a 3D
        array with 3 or 4 channels is treated as RGB/RGBA, using the first
        three channels.
    title:
        Figure title.
    show:
        Whether to call ``plt.show()`` before returning.
    figsize:
        Optional matplotlib figure size.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib. Color images return three axes; gray
        images return one axis.
    """
    arr = _to_uint8_pixels(_load_image_array(image))
    bins = np.arange(257)

    if arr.ndim == 2:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (7, 4))
        ax.hist(arr.ravel(), bins=bins, color="gray", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
        axes = ax
    else:
        colors = ("red", "green", "blue")
        labels = ("R", "G", "B")
        fig, axes = plt.subplots(1, 3, figsize=figsize or (14, 4), sharey=True)
        fig.suptitle(title)
        for channel_idx, ax in enumerate(axes):
            ax.hist(
                arr[:, :, channel_idx].ravel(),
                bins=bins,
                color=colors[channel_idx],
                alpha=0.75,
            )
            ax.set_title(f"{labels[channel_idx]} Channel")
            ax.set_xlabel("Pixel Value")
            ax.set_xlim(0, 255)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Frequency")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_correlation_analysis(
    image: ImageInput,
    *,
    sample_size: int | None = 10000,
    seed: int | None = 0,
    title: str = "Correlation Analysis",
    show_legend: bool = True,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """
    Calculate and plot adjacent-pixel correlation in three directions.

    Horizontal uses ``(x, y)`` and ``(x, y + 1)``, vertical uses ``(x, y)`` and
    ``(x + 1, y)``, and diagonal uses ``(x, y)`` and ``(x + 1, y + 1)``.
    Color images are analyzed separately for the R, G, and B channels.

    Parameters
    ----------
    image:
        Image path or numpy ndarray.
    sample_size:
        Number of adjacent pixel pairs sampled per channel and direction.
        Use ``None`` to use all available pairs.
    seed:
        Random seed used when sampling pairs. Use ``None`` for non-determinism.
    title:
        Figure title.
    show_legend:
        Whether to show a legend containing the correlation coefficients.
    show:
        Whether to call ``plt.show()`` before returning.
    figsize:
        Optional matplotlib figure size.

    Returns
    -------
    tuple
        ``(coefficients, fig, axes)``. ``coefficients`` is a nested dict like
        ``{"R": {"horizontal": 0.98, ...}, ...}``. ``axes`` is a 1D array of
        3D axes, one per channel.
    """
    arr = _to_uint8_pixels(_load_image_array(image))
    rng = np.random.default_rng(seed)
    direction_names = tuple(_DIRECTIONS)

    if arr.ndim == 2:
        channel_items = [("Gray", arr)]
    else:
        channel_items = [
            ("R", arr[:, :, 0]),
            ("G", arr[:, :, 1]),
            ("B", arr[:, :, 2]),
        ]

    coefficients: dict[str, dict[str, float]] = {}
    sampled_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for channel_name, channel in channel_items:
        coefficients[channel_name] = {}
        for direction in direction_names:
            x, y = _pixel_pairs(channel, direction)
            x_sample, y_sample = _sample_pairs(x, y, sample_size, rng)
            coefficients[channel_name][direction] = _correlation_coefficient(x_sample, y_sample)
            sampled_pairs[(channel_name, direction)] = (x_sample, y_sample)

    channel_count = len(channel_items)
    fig = plt.figure(figsize=figsize or (5.4 * channel_count, 4.4))
    fig.suptitle(title)

    axes = np.empty(channel_count, dtype=object)
    direction_colors = {
        "horizontal": "crimson",
        "vertical": "forestgreen",
        "diagonal": "dodgerblue",
    }
    direction_titles = {
        "horizontal": "Horizontal",
        "vertical": "Vertical",
        "diagonal": "Diagonal",
    }

    for channel_idx, (channel_name, _) in enumerate(channel_items):
        ax = fig.add_subplot(1, channel_count, channel_idx + 1, projection="3d")
        axes[channel_idx] = ax

        for direction_idx, direction in enumerate(direction_names):
            x_sample, y_sample = sampled_pairs[(channel_name, direction)]
            direction_axis = np.full_like(x_sample, direction_idx, dtype=np.float64)

            ax.scatter(
                direction_axis,
                x_sample,
                y_sample,
                s=3,
                c=direction_colors[direction],
                alpha=0.35,
                edgecolors="none",
                label=f"{direction_titles[direction]}: {coefficients[channel_name][direction]:.6f}",
            )

        ax.set_title(f"{channel_name} channel")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xlim(-0.35, len(direction_names) - 0.65)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.set_xticks(range(len(direction_names)))
        ax.set_xticklabels([direction_titles[name] for name in direction_names])
        ax.set_yticks([0, 100, 200])
        ax.set_zticks([0, 50, 100, 150, 200, 250])
        ax.view_init(elev=20, azim=-120)
        ax.set_box_aspect((2.2, 1.4, 1.5))
        ax.grid(visible=False)
        if show_legend:
            ax.legend(loc="upper left", fontsize=8)

    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.86, wspace=0.08)
    if show:
        plt.show()
    return coefficients, fig, axes




@dataclass
class _PipelineRun:
    """Normalized result of one encryption/decryption invocation."""

    image_path: Path
    original: np.ndarray
    encrypted: np.ndarray
    decrypted: np.ndarray
    encryption_seconds: float
    decryption_seconds: float


class Analysis:
    """Run image-encryption performance and security analyses.

    Parameters
    ----------
    encryption_function:
        A callable accepting an image path or ndarray. It may return a raw
        image, ``(encrypted_image, metadata)``, a mapping, or an object with an
        ``encrypted_image`` attribute (such as ``EncryptionResult``).
    decryption_function:
        A callable accepting the encrypted image. When the encryption result
        also contains metadata/context, it is passed as the following
        positional argument(s). The result may be a raw image, a mapping, or an
        object with a ``decrypted_image`` attribute.
    image_paths:
        Non-empty list of test image paths. Paths are validated immediately so
        a long batch does not fail half way through.

    Notes
    -----
    Every public test method executes both supplied callables. Results are
    returned as ordinary dictionaries for use in notebooks, JSON conversion,
    or further statistical processing. ``print_result=True`` also prints a
    compact human-readable summary.
    """

    def __init__(
        self,
        encryption_function: CryptFunction,
        decryption_function: CryptFunction,
        image_paths: list[str | Path],
    ) -> None:
        if not callable(encryption_function):
            raise TypeError("encryption_function must be callable")
        if not callable(decryption_function):
            raise TypeError("decryption_function must be callable")
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list")
        if not image_paths:
            raise ValueError("image_paths must contain at least one image path")

        normalized_paths = [Path(path).expanduser().resolve() for path in image_paths]
        missing_paths = [str(path) for path in normalized_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"test image does not exist: {missing_paths[0]}")

        self.encryption_function = encryption_function
        self.decryption_function = decryption_function
        self.image_paths = normalized_paths

    @staticmethod
    def _invoke(function: CryptFunction, *args: Any) -> Any:
        """Call a cryptographic function while disabling optional profiling."""
        import inspect

        quiet_kwargs: dict[str, bool] = {}
        try:
            parameters = inspect.signature(function).parameters
            if "print_profile" in parameters:
                quiet_kwargs["print_profile"] = False
            if "verbose" in parameters:
                quiet_kwargs["verbose"] = False
        except (TypeError, ValueError):
            pass
        return function(*args, **quiet_kwargs)

    @staticmethod
    def _unpack_encryption_result(result: Any) -> tuple[Any, tuple[Any, ...]]:
        if hasattr(result, "encrypted_image"):
            context = (result.metadata,) if hasattr(result, "metadata") else ()
            return result.encrypted_image, context

        if isinstance(result, Mapping):
            encrypted = next(
                (result[key] for key in ("encrypted_image", "ciphertext", "cipher", "image") if key in result),
                None,
            )
            if encrypted is None:
                raise ValueError("encryption result mapping has no encrypted image")
            context = next(
                ((result[key],) for key in ("metadata", "context", "key", "state") if key in result),
                (),
            )
            return encrypted, context

        if isinstance(result, tuple):
            if not result:
                raise ValueError("encryption function returned an empty tuple")
            return result[0], tuple(result[1:])

        return result, ()

    @staticmethod
    def _unpack_decryption_result(result: Any) -> Any:
        if hasattr(result, "decrypted_image"):
            return result.decrypted_image
        if isinstance(result, Mapping):
            for key in ("decrypted_image", "plaintext", "plain", "image"):
                if key in result:
                    return result[key]
            raise ValueError("decryption result mapping has no decrypted image")
        if isinstance(result, tuple):
            if not result:
                raise ValueError("decryption function returned an empty tuple")
            return result[0]
        return result

    def _run_pipeline(self, image: ImageInput, *, label_path: Path | None = None) -> _PipelineRun:
        original = _to_uint8_pixels(_load_image_array(image))

        started = time.perf_counter()
        encryption_result = self._invoke(self.encryption_function, image)
        encryption_seconds = time.perf_counter() - started
        encrypted_value, decryption_context = self._unpack_encryption_result(encryption_result)
        encrypted = _to_uint8_pixels(_load_image_array(encrypted_value))

        started = time.perf_counter()
        decryption_result = self._invoke(
            self.decryption_function,
            encrypted_value,
            *decryption_context,
        )
        decryption_seconds = time.perf_counter() - started
        decrypted = _to_uint8_pixels(_load_image_array(self._unpack_decryption_result(decryption_result)))

        if label_path is None:
            label_path = Path(image) if isinstance(image, (str, Path)) else Path("<array>")
        return _PipelineRun(
            image_path=label_path,
            original=original,
            encrypted=encrypted,
            decrypted=decrypted,
            encryption_seconds=encryption_seconds,
            decryption_seconds=decryption_seconds,
        )

    def _runs(self) -> list[_PipelineRun]:
        return [self._run_pipeline(path) for path in self.image_paths]

    @staticmethod
    def _fidelity(original: np.ndarray, decrypted: np.ndarray) -> dict[str, float | bool | None]:
        shape_matches = original.shape == decrypted.shape
        if not shape_matches:
            return {
                "shape_matches": False,
                "exact_recovery": False,
                "mse": None,
                "psnr_db": None,
            }
        difference = original.astype(np.float64) - decrypted.astype(np.float64)
        mse = float(np.mean(difference**2))
        psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse))
        return {
            "shape_matches": True,
            "exact_recovery": bool(np.array_equal(original, decrypted)),
            "mse": mse,
            "psnr_db": psnr,
        }

    @staticmethod
    def _histograms(image: np.ndarray) -> dict[str, np.ndarray]:
        if image.ndim == 2:
            return {"Gray": np.bincount(image.ravel(), minlength=256)}
        return {
            name: np.bincount(image[:, :, index].ravel(), minlength=256)
            for index, name in enumerate(("R", "G", "B"))
        }

    @staticmethod
    def _correlations(
        image: np.ndarray,
        sample_size: int | None,
        seed: int | None,
    ) -> dict[str, dict[str, float]]:
        rng = np.random.default_rng(seed)
        channel_items = [("Gray", image)] if image.ndim == 2 else [
            (name, image[:, :, index]) for index, name in enumerate(("R", "G", "B"))
        ]
        output: dict[str, dict[str, float]] = {}
        for channel_name, channel in channel_items:
            output[channel_name] = {}
            for direction in _DIRECTIONS:
                x, y = _pixel_pairs(channel, direction)
                x, y = _sample_pairs(x, y, sample_size, rng)
                output[channel_name][direction] = _correlation_coefficient(x, y)
        return output

    @staticmethod
    def _mean_abs_correlation(coefficients: dict[str, dict[str, float]]) -> float:
        return float(np.mean([abs(value) for channel in coefficients.values() for value in channel.values()]))

    @staticmethod
    def _entropy_summary(values: dict[str, float]) -> float:
        return values.get("average", values.get("Gray", values["overall"] if "overall" in values else 0.0))

    @staticmethod
    def _print_heading(title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")

    @staticmethod
    def _format_number(value: float | int | None, decimals: int = 6) -> str:
        if value is None:
            return "-"
        if isinstance(value, (float, np.floating)) and np.isinf(value):
            return "inf"
        return f"{float(value):.{decimals}f}"

    @staticmethod
    def _yes_no(value: Any) -> str:
        return "Yes" if bool(value) else "No"

    def test_reversibility(self, *, print_result: bool = True) -> list[dict[str, Any]]:
        """Check whether decryption reproduces every original pixel exactly."""
        results = []
        for run in self._runs():
            result = {"image": str(run.image_path), **self._fidelity(run.original, run.decrypted)}
            results.append(result)

        if print_result:
            self._print_heading("Decryption reversibility")
            _print_ascii_table(
                ("Image", "Shape Match", "Exact Recovery", "MSE", "PSNR (dB)"),
                [
                    (
                        Path(result["image"]).name,
                        self._yes_no(result["shape_matches"]),
                        self._yes_no(result["exact_recovery"]),
                        self._format_number(result["mse"]),
                        self._format_number(result["psnr_db"]),
                    )
                    for result in results
                ],
            )
        return results

    def test_encryption_decryption(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
        dpi: int = 150,
        show: bool = True,
        print_result: bool = True,
    ) -> dict[str, Any]:
        """Plot one row per image: original, encrypted, and decrypted.

        The returned ``axes`` array always has shape ``(N, 3)``, including
        when only one image is tested. The caller owns the returned figure and
        may save it again or close it with ``plt.close(result["figure"])``.
        """
        if not isinstance(dpi, int) or dpi <= 0:
            raise ValueError("dpi must be a positive integer")

        runs = self._runs()
        row_count = len(runs)
        figure, axes = plt.subplots(
            row_count,
            3,
            figsize=figsize or (12.0, 3.8 * row_count),
            squeeze=False,
            constrained_layout=True,
        )
        column_titles = ("Original", "Encrypted", "Decrypted")
        rows = []

        for row_index, run in enumerate(runs):
            fidelity = self._fidelity(run.original, run.decrypted)
            rows.append(
                {
                    "image": str(run.image_path),
                    "encryption_seconds": run.encryption_seconds,
                    "decryption_seconds": run.decryption_seconds,
                    **fidelity,
                }
            )
            for column_index, image in enumerate((run.original, run.encrypted, run.decrypted)):
                axis = axes[row_index, column_index]
                if image.ndim == 2:
                    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
                else:
                    axis.imshow(image)
                if row_index == 0:
                    axis.set_title(column_titles[column_index], fontsize=13, fontweight="bold")
                axis.axis("off")

            axes[row_index, 0].text(
                -0.04,
                0.5,
                run.image_path.name,
                transform=axes[row_index, 0].transAxes,
                rotation=90,
                ha="right",
                va="center",
                fontsize=10,
            )

        saved_path: Path | None = None
        if save_path is not None:
            saved_path = Path(save_path).expanduser().resolve()
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(saved_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        if print_result:
            self._print_heading("Encryption/decryption image comparison")
            _print_ascii_table(
                (
                    "Image",
                    "Encrypt (s)",
                    "Decrypt (s)",
                    "Exact Recovery",
                    "MSE",
                    "PSNR (dB)",
                    "Saved Figure",
                ),
                [
                    (
                        Path(row["image"]).name,
                        self._format_number(row["encryption_seconds"]),
                        self._format_number(row["decryption_seconds"]),
                        self._yes_no(row["exact_recovery"]),
                        self._format_number(row["mse"]),
                        self._format_number(row["psnr_db"]),
                        str(saved_path) if saved_path is not None and index == 0 else "-",
                    )
                    for index, row in enumerate(rows)
                ],
            )

        if show:
            plt.show()
        return {
            "figure": figure,
            "axes": axes,
            "rows": rows,
            "save_path": str(saved_path) if saved_path is not None else None,
        }

    def test_speed(
        self,
        *,
        repeats: int = 3,
        warmup: int = 0,
        print_result: bool = True,
    ) -> list[dict[str, Any]]:
        """Measure end-to-end encryption/decryption latency and throughput."""
        if not isinstance(repeats, int) or repeats <= 0:
            raise ValueError("repeats must be a positive integer")
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup must be a non-negative integer")

        results = []
        for path in self.image_paths:
            for _ in range(warmup):
                self._run_pipeline(path)
            runs = [self._run_pipeline(path) for _ in range(repeats)]
            encryption_times = np.array([run.encryption_seconds for run in runs])
            decryption_times = np.array([run.decryption_seconds for run in runs])
            pixel_count = runs[0].original.shape[0] * runs[0].original.shape[1]
            encryption_mean = float(encryption_times.mean())
            decryption_mean = float(decryption_times.mean())
            results.append(
                {
                    "image": str(path),
                    "shape": tuple(int(value) for value in runs[0].original.shape),
                    "repeats": repeats,
                    "encryption_seconds_mean": encryption_mean,
                    "encryption_seconds_std": float(encryption_times.std(ddof=0)),
                    "decryption_seconds_mean": decryption_mean,
                    "decryption_seconds_std": float(decryption_times.std(ddof=0)),
                    "encryption_mpixel_per_second": float(pixel_count / encryption_mean / 1e6),
                    "decryption_mpixel_per_second": float(pixel_count / decryption_mean / 1e6),
                    "exact_recovery": all(np.array_equal(run.original, run.decrypted) for run in runs),
                }
            )

        if print_result:
            self._print_heading("Encryption/decryption speed")
            _print_ascii_table(
                (
                    "Image",
                    "Shape",
                    "Runs",
                    "Enc Mean (s)",
                    "Enc Std (s)",
                    "Enc MPix/s",
                    "Dec Mean (s)",
                    "Dec Std (s)",
                    "Dec MPix/s",
                    "Exact Recovery",
                ),
                [
                    (
                        Path(result["image"]).name,
                        "x".join(str(value) for value in result["shape"]),
                        result["repeats"],
                        self._format_number(result["encryption_seconds_mean"]),
                        self._format_number(result["encryption_seconds_std"]),
                        self._format_number(result["encryption_mpixel_per_second"], 3),
                        self._format_number(result["decryption_seconds_mean"]),
                        self._format_number(result["decryption_seconds_std"]),
                        self._format_number(result["decryption_mpixel_per_second"], 3),
                        self._yes_no(result["exact_recovery"]),
                    )
                    for result in results
                ],
            )
        return results

    def test_histogram(
        self,
        *,
        plot: bool = False,
        show: bool = False,
        print_result: bool = True,
    ) -> list[dict[str, Any]]:
        """Compare original, encrypted, and decrypted 256-bin histograms."""
        results = []
        for run in self._runs():
            figures = None
            if plot:
                figures = {}
                for name, image in (
                    ("original", run.original),
                    ("encrypted", run.encrypted),
                    ("decrypted", run.decrypted),
                ):
                    figure, _ = plot_pixel_histogram(
                        image,
                        title=f"{run.image_path.name} - {name}",
                        show=show,
                    )
                    figures[name] = figure
            results.append(
                {
                    "image": str(run.image_path),
                    "original": self._histograms(run.original),
                    "encrypted": self._histograms(run.encrypted),
                    "decrypted": self._histograms(run.decrypted),
                    "figures": figures,
                    "exact_recovery": bool(np.array_equal(run.original, run.decrypted)),
                }
            )

        if print_result:
            self._print_heading("Histogram analysis")
            table_rows: list[tuple[Any, ...]] = []
            for result in results:
                channel_cvs = {
                    channel: float(counts.std() / counts.mean())
                    for channel, counts in result["encrypted"].items()
                }
                for channel, coefficient_of_variation in channel_cvs.items():
                    table_rows.append(
                        (
                            Path(result["image"]).name,
                            channel,
                            self._format_number(coefficient_of_variation),
                            self._yes_no(result["exact_recovery"]),
                        )
                    )
                table_rows.append(
                    (
                        Path(result["image"]).name,
                        "Average",
                        self._format_number(float(np.mean(list(channel_cvs.values())))),
                        self._yes_no(result["exact_recovery"]),
                    )
                )
            _print_ascii_table(
                ("Image", "Channel", "Encrypted Histogram CV", "Exact Recovery"),
                table_rows,
            )
        return results

    def test_entropy(self, *, print_result: bool = True) -> list[dict[str, Any]]:
        """Calculate Shannon entropy; encrypted-channel values should approach 8."""
        results = []
        for run in self._runs():
            original = calculate_information_entropy(run.original)
            encrypted = calculate_information_entropy(run.encrypted)
            decrypted = calculate_information_entropy(run.decrypted)
            encrypted_summary = self._entropy_summary(encrypted)
            results.append(
                {
                    "image": str(run.image_path),
                    "original": original,
                    "encrypted": encrypted,
                    "decrypted": decrypted,
                    "encrypted_average": encrypted_summary,
                    "distance_from_ideal": 8.0 - encrypted_summary,
                    "exact_recovery": bool(np.array_equal(run.original, run.decrypted)),
                }
            )

        if print_result:
            self._print_heading("Information entropy")
            table_rows = []
            for result in results:
                channels = [
                    channel
                    for channel in ("R", "G", "B", "Gray", "average", "overall")
                    if channel in result["encrypted"]
                ]
                for channel in channels:
                    table_rows.append(
                        (
                            Path(result["image"]).name,
                            channel.title(),
                            self._format_number(result["original"].get(channel)),
                            self._format_number(result["encrypted"][channel]),
                            self._format_number(result["decrypted"].get(channel)),
                            self._format_number(8.0 - result["encrypted"][channel]),
                            self._yes_no(result["exact_recovery"]),
                        )
                    )
            _print_ascii_table(
                (
                    "Image",
                    "Channel",
                    "Original (bit)",
                    "Encrypted (bit)",
                    "Decrypted (bit)",
                    "Distance to 8",
                    "Exact Recovery",
                ),
                table_rows,
            )
        return results

    def test_chi_square(
        self,
        *,
        alpha: float = 0.05,
        critical_value: float | None = None,
        print_result: bool = True,
    ) -> list[dict[str, Any]]:
        """Test whether ciphertext histograms are consistent with uniformity."""
        results = []
        for run in self._runs():
            original = calculate_chi_square_test(
                run.original, alpha=alpha, critical_value=critical_value, print_result=False
            )
            encrypted = calculate_chi_square_test(
                run.encrypted, alpha=alpha, critical_value=critical_value, print_result=False
            )
            decrypted = calculate_chi_square_test(
                run.decrypted, alpha=alpha, critical_value=critical_value, print_result=False
            )
            results.append(
                {
                    "image": str(run.image_path),
                    "original": original,
                    "encrypted": encrypted,
                    "decrypted": decrypted,
                    "encrypted_passed": all(
                        bool(channel_result["passed"])
                        for channel, channel_result in encrypted.items()
                        if channel != "average"
                    ),
                    "exact_recovery": bool(np.array_equal(run.original, run.decrypted)),
                }
            )

        if print_result:
            self._print_heading("Chi-square histogram uniformity")
            effective_critical_value = 293.2478 if critical_value is None else critical_value
            channel_order = [
                channel
                for channel in ("R", "G", "B", "Gray")
                if any(channel in result["encrypted"] for result in results)
            ]
            headers = (
                "Image",
                *(f"{channel} Chi-square" for channel in channel_order),
                "Average Chi-square",
                "Critical Value",
                "Alpha",
                "Uniformity",
                "Exact Recovery",
            )
            table_rows = []
            for result in results:
                table_rows.append(
                    (
                        Path(result["image"]).name,
                        *(
                            self._format_number(
                                result["encrypted"][channel]["chi_square"]
                                if channel in result["encrypted"]
                                else None,
                                4,
                            )
                            for channel in channel_order
                        ),
                        self._format_number(result["encrypted"]["average"]["chi_square"], 4),
                        self._format_number(effective_critical_value, 4),
                        self._format_number(alpha, 4),
                        "Pass" if result["encrypted_passed"] else "Fail",
                        self._yes_no(result["exact_recovery"]),
                    )
                )
            _print_ascii_table(headers, table_rows)
        return results

    def test_correlation(
        self,
        *,
        sample_size: int | None = 10000,
        seed: int | None = 0,
        plot: bool = False,
        show: bool = False,
        print_result: bool = True,
    ) -> list[dict[str, Any]]:
        """Analyze horizontal, vertical, and diagonal adjacent-pixel correlation."""
        if sample_size is not None and sample_size <= 0:
            raise ValueError("sample_size must be positive or None")

        results = []
        for run in self._runs():
            original = self._correlations(run.original, sample_size, seed)
            encrypted = self._correlations(run.encrypted, sample_size, seed)
            decrypted = self._correlations(run.decrypted, sample_size, seed)
            figures = None
            if plot:
                figures = {}
                for name, image in (
                    ("original", run.original),
                    ("encrypted", run.encrypted),
                    ("decrypted", run.decrypted),
                ):
                    _, figure, _ = plot_correlation_analysis(
                        image,
                        sample_size=sample_size,
                        seed=seed,
                        title=f"{run.image_path.name} - {name}",
                        show=show,
                    )
                    figures[name] = figure
            results.append(
                {
                    "image": str(run.image_path),
                    "original": original,
                    "encrypted": encrypted,
                    "decrypted": decrypted,
                    "encrypted_mean_absolute": self._mean_abs_correlation(encrypted),
                    "figures": figures,
                    "exact_recovery": bool(np.array_equal(run.original, run.decrypted)),
                }
            )

        if print_result:
            self._print_heading("Adjacent-pixel correlation")
            table_rows = []
            for result in results:
                direction_values = {direction: [] for direction in _DIRECTIONS}
                for channel, directions in result["encrypted"].items():
                    for direction, value in directions.items():
                        direction_values[direction].append(value)
                    mean_absolute = float(np.mean([abs(value) for value in directions.values()]))
                    table_rows.append(
                        (
                            Path(result["image"]).name,
                            channel,
                            self._format_number(directions["horizontal"]),
                            self._format_number(directions["vertical"]),
                            self._format_number(directions["diagonal"]),
                            self._format_number(mean_absolute),
                            self._yes_no(result["exact_recovery"]),
                        )
                    )
                table_rows.append(
                    (
                        Path(result["image"]).name,
                        "Average",
                        self._format_number(float(np.mean(direction_values["horizontal"]))),
                        self._format_number(float(np.mean(direction_values["vertical"]))),
                        self._format_number(float(np.mean(direction_values["diagonal"]))),
                        self._format_number(result["encrypted_mean_absolute"]),
                        self._yes_no(result["exact_recovery"]),
                    )
                )
            _print_ascii_table(
                (
                    "Image",
                    "Channel",
                    "Horizontal r",
                    "Vertical r",
                    "Diagonal r",
                    "Mean Abs(r)",
                    "Exact Recovery",
                ),
                table_rows,
            )
        return results

    def test_differential_attack(self, *, print_result: bool = True) -> list[dict[str, Any]]:
        """Measure NPCR/UACI after changing one plaintext pixel by one bit."""
        results = []
        for path in self.image_paths:
            original = _to_uint8_pixels(_load_image_array(path))
            modified = original.copy()
            modified.reshape(-1)[0] ^= np.uint8(1)

            baseline_run = self._run_pipeline(original, label_path=path)
            modified_run = self._run_pipeline(modified, label_path=path)
            if baseline_run.encrypted.shape != modified_run.encrypted.shape:
                raise ValueError("ciphertext shape changed after a one-bit plaintext modification")

            cipher_a = baseline_run.encrypted.astype(np.int16)
            cipher_b = modified_run.encrypted.astype(np.int16)
            npcr = float(np.mean(cipher_a != cipher_b) * 100.0)
            uaci = float(np.mean(np.abs(cipher_a - cipher_b)) / 255.0 * 100.0)
            results.append(
                {
                    "image": str(path),
                    "npcr_percent": npcr,
                    "uaci_percent": uaci,
                    "npcr_near_ideal": npcr >= 99.0,
                    "uaci_near_ideal": 30.0 <= uaci <= 36.0,
                    "baseline_exact_recovery": bool(
                        np.array_equal(baseline_run.original, baseline_run.decrypted)
                    ),
                    "modified_exact_recovery": bool(
                        np.array_equal(modified_run.original, modified_run.decrypted)
                    ),
                }
            )

        if print_result:
            self._print_heading("Differential attack (one-bit plaintext change)")
            _print_ascii_table(
                (
                    "Image",
                    "NPCR (%)",
                    "NPCR Near Ideal",
                    "UACI (%)",
                    "UACI Near Ideal",
                    "Baseline Exact",
                    "Modified Exact",
                ),
                [
                    (
                        Path(result["image"]).name,
                        self._format_number(result["npcr_percent"]),
                        self._yes_no(result["npcr_near_ideal"]),
                        self._format_number(result["uaci_percent"]),
                        self._yes_no(result["uaci_near_ideal"]),
                        self._yes_no(result["baseline_exact_recovery"]),
                        self._yes_no(result["modified_exact_recovery"]),
                    )
                    for result in results
                ],
            )
        return results

    def run_all(
        self,
        *,
        speed_repeats: int = 3,
        include_differential: bool = True,
        include_image_comparison: bool = True,
        show_image_comparison: bool = False,
        print_result: bool = True,
    ) -> dict[str, Any]:
        """Run all analyses and return one result mapping."""
        results: dict[str, Any] = {
            "reversibility": self.test_reversibility(print_result=print_result),
            "speed": self.test_speed(repeats=speed_repeats, print_result=print_result),
            "histogram": self.test_histogram(print_result=print_result),
            "entropy": self.test_entropy(print_result=print_result),
            "chi_square": self.test_chi_square(print_result=print_result),
            "correlation": self.test_correlation(print_result=print_result),
        }
        if include_differential:
            results["differential"] = self.test_differential_attack(print_result=print_result)
        if include_image_comparison:
            results["image_comparison"] = self.test_encryption_decryption(
                show=show_image_comparison,
                print_result=print_result,
            )
        return results

    # Natural aliases for code written with the old ``*_test`` naming style.
    reversibility_test = test_reversibility
    encryption_decryption_test = test_encryption_decryption
    speed_test = test_speed
    histogram_test = test_histogram
    entropy_test = test_entropy
    chi_square_test = test_chi_square
    correlation_test = test_correlation
    differential_attack_test = test_differential_attack


__all__ = [
    "Analysis",
    "ImageInput",
    "calculate_chi_square_test",
    "calculate_information_entropy",
    "plot_correlation_analysis",
    "plot_pixel_histogram",
]


if __name__ == "__main__":
    print("1")
    from .Encryption import (
    Encrypter,
    DeEncrypter,
    EncryptionConfig,)
    
    config = EncryptionConfig(
    seed=2026,
    b_max=64,
    b_min=12,
    block_operation="xor",
    global_parallel_size=64,)
    MyEncrypter = Encrypter(config)
    MyDeEncrypter = DeEncrypter(config)
    
    analysis = Analysis(MyEncrypter.encrypt, MyDeEncrypter.decrypt,
                        [
                        # r"C:\ImageEncryption\images\img1.png",
                        # r"C:\ImageEncryption\images\img2.png",
                        # r"C:\ImageEncryption\images\img3.png",
                        # r"C:\ImageEncryption\images\img4.png",
                        r"C:\ImageEncryption\images\img5.png",
                        r"C:\ImageEncryption\images\img6.png",
                        r"C:\ImageEncryption\images\img7.png",
                        ])
    result = analysis.encryption_decryption_test(show=True)
    #卡方检验 global_parallel_size居然会影响卡方检验的结果，取64不错
    analysis.chi_square_test()
    #直方图
    # analysis.histogram_test(plot=True, show=True)
    #相关性分析
    analysis.test_correlation(plot=False, show=True)
    #信息熵
    analysis.entropy_test()
    #差分攻击
    analysis.differential_attack_test()
