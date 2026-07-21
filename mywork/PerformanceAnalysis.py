from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_image
from .ScrambleDiffusion import encrypt_image


ImageInput = Union[str, Path, np.ndarray]

_DIRECTIONS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diagonal": (1, 1),
}


def _load_image_array(image: ImageInput) -> np.ndarray:
    """Load a path or ndarray into a grayscale or RGB ndarray."""
    if isinstance(image, (str, Path)):
        with pil_image.open(image) as img:
            if len(img.getbands()) == 1:
                return np.asarray(img.convert("L"))
            return np.asarray(img.convert("RGB"))

    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                return arr[:, :, 0]
            return arr[:, :, :3]
        raise ValueError("ndarray image must be 2D grayscale or 3D with 1, 3, or 4 channels")

    raise TypeError("image must be a file path or numpy ndarray")


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
        print(f"Chi-square test (alpha={alpha}, critical_value={critical_value:.4f})")
        print(f"{'Channel':<10}{'Chi-square':>15}{'Result':>10}")
        for channel_name, result in results.items():
            mark = "√" if result["passed"] else "×"
            print(f"{channel_name:<10}{result['chi_square']:>15.4f}{mark:>10}")

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




def histogram_test():
    img1_path = Path(r"C:\ImageEncryption\images\img1.png")
    plot_pixel_histogram(img1_path, title="")
    encrypted_img = encrypt_image(img1_path, verbose=False)
    plot_pixel_histogram(encrypted_img, title="")
    img2_path = Path(r"C:\ImageEncryption\images\img2.png")
    plot_pixel_histogram(img2_path, title="")
    encrypted_img2 = encrypt_image(img2_path, verbose=False)
    plot_pixel_histogram(encrypted_img2, title="")
    img3_path = Path(r"C:\ImageEncryption\images\img5.png")
    plot_pixel_histogram(img3_path, title="")
    encrypted_img3 = encrypt_image(img3_path, verbose=False)
    plot_pixel_histogram(encrypted_img3, title="")


def correlation_test():
    img1_path = Path(r"C:\ImageEncryption\image\img1.png")
    coeffs, fig, axes = plot_correlation_analysis(img1_path)
    encrypted_img1 = encrypt_image(img1_path, verbose=False)
    coeffs_enc, fig_enc, axes_enc = plot_correlation_analysis(encrypted_img1)
    
    img2_path = Path(r"C:\ImageEncryption\image\img2.png")
    coeffs2, fig2, axes2 = plot_correlation_analysis(img2_path)
    encrypted_img2 = encrypt_image(img2_path, verbose=False)
    coeffs2_enc, fig2_enc, axes2_enc = plot_correlation_analysis(encrypted_img2)

    img3_path = Path(r"C:\ImageEncryption\image\img3.png")
    coeffs3, fig3, axes3 = plot_correlation_analysis(img3_path)
    encrypted_img3 = encrypt_image(img3_path, verbose=False)
    coeffs3_enc, fig3_enc, axes3_enc = plot_correlation_analysis(encrypted_img3)


def entropy_test():
    img1_path = Path(r"C:\ImageEncryption\image\img1.png")
    print("Image 1 entropy:", calculate_information_entropy(img1_path))
    encrypted_img1 = encrypt_image(img1_path, verbose=False)
    print("Encrypted Image 1 entropy:", calculate_information_entropy(encrypted_img1))

    img2_path = Path(r"C:\ImageEncryption\image\img2.png")
    print("Image 2 entropy:", calculate_information_entropy(img2_path))
    encrypted_img2 = encrypt_image(img2_path, verbose=False)
    print("Encrypted Image 2 entropy:", calculate_information_entropy(encrypted_img2))

    img3_path = Path(r"C:\ImageEncryption\image\img3.png")
    print("Image 3 entropy:", calculate_information_entropy(img3_path))
    encrypted_img3 = encrypt_image(img3_path, verbose=False)
    print("Encrypted Image 3 entropy:", calculate_information_entropy(encrypted_img3))
    
def chi_square_test():
    img1_path = Path(r"C:\ImageEncryption\images\img1.png")
    encrypted_img1 = encrypt_image(img1_path, verbose=False)
    calculate_chi_square_test(encrypted_img1)
    
    img2_path = Path(r"C:\ImageEncryption\images\img2.png")
    encrypted_img2 = encrypt_image(img2_path, verbose=False)
    calculate_chi_square_test(encrypted_img2)
    
    img3_path = Path(r"C:\ImageEncryption\images\random_noise.png")
    encrypted_img3 = encrypt_image(img3_path, verbose=False)
    calculate_chi_square_test(encrypted_img3)
    
def encrypt_test():
    img1_path = Path(r"C:\ImageEncryption\image\img1.png")
    encrypted_img1 = encrypt_image(img1_path, verbose=False)
    #save to the path
    img = pil_image.fromarray(encrypted_img1)
    img.save(r"C:\ImageEncryption\image\encrypted_img1.png")
    
    img2_path = Path(r"C:\ImageEncryption\image\img2.png")
    encrypted_img2 = encrypt_image(img2_path, verbose=False)
    img = pil_image.fromarray(encrypted_img2)
    img.save(r"C:\ImageEncryption\image\encrypted_img2.png")
    
    img3_path = Path(r"C:\ImageEncryption\image\img3.png")
    encrypted_img3 = encrypt_image(img3_path, verbose=False)
    img = pil_image.fromarray(encrypted_img3)
    img.save(r"C:\ImageEncryption\image\encrypted_img3.png")
    
    
if __name__ == "__main__":
    img1_path = Path(r"C:\ImageEncryption\images\img3.png")
    # histogram_test()
    # correlation_test()
    # entropy_test()
    chi_square_test()
    # encrypt_test()
