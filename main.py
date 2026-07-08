import sys
from pathlib import Path
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import numpy as np
import time

PROJECT_ROOT = Path(r"C:\ImageEncryptionV2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "papers" / "stablediffusion"))

from papers.stablediffusion.Mytest import *


def vis_distribution(
    matrix,
    bins: int = 100,
    title: str | None = None,
    show: bool = True,
    robust_percentile: float = 0.5,
):
    """Plot value distributions for a tensor or ndarray with robust histogram bounds."""
    if torch.is_tensor(matrix):
        array = matrix.detach().cpu().numpy()
        source_name = "torch.Tensor"
    elif isinstance(matrix, np.ndarray):
        array = np.asarray(matrix)
        source_name = "numpy.ndarray"
    else:
        raise TypeError("matrix must be a torch.Tensor or numpy.ndarray.")

    if array.size == 0:
        raise ValueError("matrix is empty.")
    if not (0.0 <= robust_percentile < 50.0):
        raise ValueError("robust_percentile must be in [0, 50).")

    array = np.asarray(array)
    title = title or f"Matrix Distribution ({source_name}, shape={array.shape})"

    def _prepare_hist_values(x: np.ndarray):
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(flat)
        finite_values = flat[finite_mask]
        nonfinite_count = int((~finite_mask).sum())
        if finite_values.size == 0:
            return finite_values, nonfinite_count, 0, None, {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "count": 0,
            }

        raw_min = float(finite_values.min())
        raw_max = float(finite_values.max())
        raw_mean = float(finite_values.mean())
        raw_stats = {
            "min": raw_min,
            "max": raw_max,
            "mean": raw_mean,
            "count": int(finite_values.size),
        }
        if raw_min == raw_max:
            return finite_values, nonfinite_count, 0, (raw_min, raw_max), raw_stats

        if robust_percentile > 0.0:
            low = float(np.percentile(finite_values, robust_percentile))
            high = float(np.percentile(finite_values, 100.0 - robust_percentile))
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = raw_min, raw_max
        else:
            low, high = raw_min, raw_max

        clipped_values = np.clip(finite_values, low, high)
        clipped_count = int(np.count_nonzero((finite_values < low) | (finite_values > high)))
        return clipped_values, nonfinite_count, clipped_count, (low, high), raw_stats

    def _draw_hist(ax, values, nonfinite_count, clipped_count, hist_range, raw_stats, label, color):
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_title(f"{label} | nonfinite={nonfinite_count}")
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            return

        if hist_range is not None and hist_range[0] == hist_range[1]:
            center = hist_range[0]
            ax.axvline(center, color=color, linewidth=2)
            ax.set_xlim(center - 1.0, center + 1.0)
        else:
            ax.hist(values, bins=bins, range=hist_range, color=color, alpha=0.8, edgecolor="black")

        title = (
            f"{label} | count={raw_stats['count']} | min={raw_stats['min']:.4g} | "
            f"max={raw_stats['max']:.4g} | mean={raw_stats['mean']:.4g} | "
            f"clipped={clipped_count} | nonfinite={nonfinite_count}"
        )
        if clipped_count > 0 and hist_range is not None:
            title += f" | hist_range=[{hist_range[0]:.4g}, {hist_range[1]:.4g}]"
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    is_latent_4ch = array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == 4

    if is_latent_4ch:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.reshape(-1)
        fig.suptitle(title)

        for channel_idx in range(4):
            channel_values, nonfinite_count, clipped_count, hist_range, raw_stats = _prepare_hist_values(array[0, channel_idx])
            _draw_hist(
                axes[channel_idx],
                channel_values,
                nonfinite_count,
                clipped_count,
                hist_range,
                raw_stats,
                f"channel {channel_idx}",
                f"C{channel_idx}",
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        finite_values, nonfinite_count, clipped_count, hist_range, raw_stats = _prepare_hist_values(array)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        _draw_hist(ax, finite_values, nonfinite_count, clipped_count, hist_range, raw_stats, title, "steelblue")
        plt.tight_layout()

    if show:
        plt.show()

    return fig


class ImageEncrypter:
    def __init__(self, image):
        if isinstance(image, str):
            self.image = pil_image.open(image)
        elif isinstance(image, pil_image.Image.Image):
            self.image = image
        else:
            raise ValueError("Unsupported image type. ")
        self.image = self.image.resize((256, 256), pil_image.LANCZOS)
        self.w = self.image.width
        self.h = self.image.height
        self.last_plain_latent = None
        self.last_noise_maps = None
        self.last_encrypted_latent = None
        init_webui_env()

    @staticmethod
    def _ensure_matching_latent_tensors(latent, noise_map):
        if not torch.is_tensor(latent) or not torch.is_tensor(noise_map):
            raise TypeError("latent and noise_map must be torch.Tensor.")
        if latent.shape != noise_map.shape:
            raise ValueError(f"Shape mismatch: latent={tuple(latent.shape)}, noise_map={tuple(noise_map.shape)}")
        if latent.ndim != 4:
            raise ValueError(f"latent must have shape [B, C, H, W], got {tuple(latent.shape)}")
        latent = latent.detach().clone().to(dtype=torch.float32)
        noise_map = noise_map.detach().clone().to(device=latent.device, dtype=torch.float32)
        return latent, noise_map

    @staticmethod
    def _xor_float32_views(*tensors):
        if len(tensors) == 0:
            raise ValueError("At least one tensor is required for XOR.")
        result = tensors[0].contiguous().view(torch.int32).clone()
        for tensor in tensors[1:]:
            result = torch.bitwise_xor(result, tensor.contiguous().view(torch.int32))
        return result.view(torch.float32)

    @staticmethod
    def diffusion_step_v1(latent, noise_map):
        latent, noise_map = ImageEncrypter._ensure_matching_latent_tensors(latent, noise_map)
        return latent + noise_map

    @staticmethod
    def dediffusion_step_v1(latent, noise_map):
        latent, noise_map = ImageEncrypter._ensure_matching_latent_tensors(latent, noise_map)
        return latent - noise_map





    def encrypt(self, prompt="a cat", negative_prompt="", seed=6, steps=20, cfg_scale=7.0):

        latent = encode_image_to_latent(self.image, vis=False)
        # vis_distribution(latent, title="Initial Latent Distribution")
        self.last_plain_latent = latent.clone().detach().to(dtype=torch.float32, device="cpu")
        st = time.time()
        noise_maps = gen_random_noise_maps(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            width=self.w,
            height=self.h,
            cfg_scale=cfg_scale,
            batch_size=1,
        )
        print(f"Noise maps generated and cached in {time.time() - st:.2f} seconds.")
        self.last_noise_maps = [noise.detach().clone().to(dtype=torch.float32, device="cpu") for noise in noise_maps]

        st = time.time()
        diffusioned_latent = self.DiffusionProcess(latent, noise_maps)
        print(f"DiffusionProcess completed in {time.time() - st:.4f} seconds.")
        dediffusioned_latent = self.DeDiffusionProcess(diffusioned_latent, noise_maps)
        max_diff = torch.abs(latent - dediffusioned_latent).max().item()
        print(f"Max absolute difference after dediffusion: {max_diff:.6f}")
        vis_latent(diffusioned_latent)
        vis_latent(dediffusioned_latent)
        vis_latent(self.last_plain_latent)
        
        
        vis_distribution(diffusioned_latent, title="Diffusioned Latent Distribution")
        self.diffusioned_latent = diffusioned_latent.clone().detach().to(dtype=torch.float32, device="cpu")

        #TODO diffusioned_latent 进行一次置乱
        shuffled_latent = diffusioned_latent.clone()
        # 将 shuffled_latent 解码为密文
        st = time.time()
        encrypted_image = decode_latent_to_image(shuffled_latent)
        print(f"Decoded image in {time.time() - st:.4f} seconds.")
        
        plt.imshow(encrypted_image)
        plt.show()

        return diffusioned_latent

    def decrypt(self, encrypted_image=None, noise_maps=None, vis: bool = False):
        encrypted_latent = encrypted_image
        if encrypted_latent is None:
            encrypted_latent = self.last_encrypted_latent
        if encrypted_latent is None:
            raise ValueError("encrypted_image is required when no cached encrypted latent is available.")

        if noise_maps is None:
            noise_maps = self.last_noise_maps
        if noise_maps is None:
            raise ValueError("noise_maps is required when no cached noise maps are available.")

        restored_latent = self.DeDiffusionProcess(encrypted_latent, noise_maps)
        if vis:
            vis_latent(restored_latent, model="grayscale")
        return restored_latent

    def DiffusionProcess(self, latent, noise_maps):
        result = latent.clone().to(dtype=torch.float32)
        count = 1
        for noise_map in noise_maps:

            vis_latent(result, title=f"Diffusion Step {count} - Before", model="value_color")
            vis_distribution(result, title=f"Diffusion Step {count} - Before", show=True)
                # vis_latent(noise_map, model="value_color")
                # vis_distribution(noise_map, title=f"Noise Map - Step {count}", show=True)
            
            result = self.diffusion_step_v2(result, noise_map)
            count += 1
        return result

    @staticmethod
    def diffusion_step_v2(latent, noise_map):
        latent, noise_map = ImageEncrypter._ensure_matching_latent_tensors(latent, noise_map)
        width = latent.shape[-1]
        mid = width // 2
        result = latent.clone()
        
        noise_mean = noise_map.mean()
        noise_std = noise_map.std()
        latent_mean = latent.mean()
        latent_std = latent.std()
        #归一化再重构：(X - μ_noise) / σ_noise * σ_latent + μ_latent
        eps = 1e-6  # 防止除以 0
        noise_map_norm = ((noise_map - noise_mean) / (noise_std + eps)) * latent_std + latent_mean

        result[..., mid] = result[..., mid] + noise_map_norm[..., mid]

        for y in range(mid - 1, -1, -1):
            if y == 2:
                vis_latent(result, title=f"Diffusion Step - Processing column {y} (left side)", model="value_color")
                # vis_latent(noise_map, model="value_color")
                vis_distribution(result, title=f"Diffusion Step - Processing column {y} (left side)", show=True)
                vis_distribution(noise_map_norm, title=f"Noise Map - Column {y} (left side)", show=True)
            result[..., y] = result[..., y + 1] + result[..., y] - noise_map_norm[..., y]

        for y in range(mid + 1, width):
            # if y == width-1:
            #     vis_latent(result, title=f"Diffusion Step - Processing column {y} (right side)", model="value_color")
            #     # vis_latent(noise_map, model="value_color")
            #     vis_distribution(result, title=f"Diffusion Step - Processing column {y} (right side)", show=True)
            #     vis_distribution(noise_map, title=f"Noise Map - Column {y} (right side)", show=True)
            result[..., y] = result[..., y - 1] + result[..., y] - noise_map_norm[..., y]

        return result
    
    def DiffusionProcess_v3(self, latent, noise_maps):
        """A more complex diffusion process that incorporates neighboring columns in a non-linear way."""
        result = latent.clone().to(dtype=torch.float32)
        width = latent.shape[-1]
        mid = width // 2
        
    
    def DeDiffusionProcess(self, diffusioned_latent, noise_maps):
        """Reverse DiffusionProcess with mathematically correct step order."""
        result = diffusioned_latent.clone().to(dtype=torch.float32)
        for noise_map in reversed(noise_maps):
            result = self.dediffusion_step_v2(result, noise_map)
        return result
    
    @staticmethod
    def dediffusion_step_v2(cipher, noise_map):
        cipher, noise_map = ImageEncrypter._ensure_matching_latent_tensors(cipher, noise_map)
        width = cipher.shape[-1]
        mid = width // 2
        plain = cipher.clone()

        plain[..., mid] = cipher[..., mid] - noise_map[..., mid]

        for y in range(mid - 1, -1, -1):
            plain[..., y] = cipher[..., y] - cipher[..., y + 1] + noise_map[..., y]

        for y in range(mid + 1, width):
            plain[..., y] = cipher[..., y] - cipher[..., y - 1] + noise_map[..., y]

        return plain




if __name__ == "__main__":
    timg_1_pth = r"C:\ImageEncryptionV2\image\img1.png"
    timg_2_pth = r"C:\ImageEncryptionV2\image\img2.png"
    timg_3_pth = r"C:\ImageEncryptionV2\papers\stablediffusion\images\test.png"
    e = ImageEncrypter(timg_1_pth)
    e.encrypt()
