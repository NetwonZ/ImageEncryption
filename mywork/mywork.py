import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
from salomon import SalomoncouplingCML
import time

# demo_1()主要测试将一个图片分解为(24,w,h)的数组，分别对R,G,B通道的高6位进行打乱，检验效果如何
# 加密方案如下：拿出某一张位平面，随机生成一个(w,h)的随机数矩阵，然后对这个随机数矩阵进行排序得到S。按照这个S的顺序对位平面进行重排，得到加密后的位平面。解密时按照S的顺序对加密后的位平面进行重排。
# 未参与加密的位平面保持不变，直接放回原来的位置。
# 结论--效果不错！
def demo_1():
    def _image_to_bitplanes(img_arr: np.ndarray) -> np.ndarray:
        h, w, c = img_arr.shape
        if c != 3:
            raise ValueError("Only RGB images are supported")

        bitplanes = np.empty((24, h, w), dtype=np.uint8)
        plane_idx = 0
        for channel in range(3):
            for bit in range(7, -1, -1):
                bitplanes[plane_idx] = (img_arr[:, :, channel] >> bit) & 1
                plane_idx += 1
        return bitplanes


    def _bitplanes_to_image(bitplanes: np.ndarray) -> np.ndarray:
        if bitplanes.shape[0] != 24:
            raise ValueError("bitplanes shape must be (24, h, w)")

        h, w = bitplanes.shape[1:]
        img_arr = np.zeros((h, w, 3), dtype=np.uint8)
        plane_idx = 0
        for channel in range(3):
            channel_data = np.zeros((h, w), dtype=np.uint8)
            for bit in range(7, -1, -1):
                channel_data |= bitplanes[plane_idx].astype(np.uint8) << bit
                plane_idx += 1
            img_arr[:, :, channel] = channel_data
        return img_arr


    def _permute_plane(plane: np.ndarray, order: np.ndarray) -> np.ndarray:
        flat_plane = plane.reshape(-1)
        return flat_plane[order].reshape(plane.shape)


    def _inverse_permute_plane(plane: np.ndarray, order: np.ndarray) -> np.ndarray:
        flat_plane = plane.reshape(-1)
        restored = np.empty_like(flat_plane)
        restored[order] = flat_plane
        return restored.reshape(plane.shape)


    def _show_demo_results(
        original: np.ndarray,
        encrypted: np.ndarray,
        decrypted: np.ndarray,
        changed_mask: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(encrypted)
        axes[0, 1].set_title("Encrypted (high 6 bits shuffled)")
        axes[1, 0].imshow(decrypted)
        axes[1, 0].set_title("Decrypted")
        axes[1, 1].imshow(changed_mask, cmap="gray")
        axes[1, 1].set_title("Changed pixels")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    img_pth = r"C:\ImageEncryptionV2\image\img1.png"
    img = pil_image.open(img_pth).convert("RGB")
    img_arr = np.asarray(img, dtype=np.uint8)


    bitplanes = _image_to_bitplanes(img_arr)
    encrypted_bitplanes = bitplanes.copy()
    decrypted_bitplanes = bitplanes.copy()

    h, w = img_arr.shape[:2]
    L = w*h
    params = {
        "mu": 5,
        "lam": 5,
        "a": 100,
        "b": 200,
        "xi": 1,
        "eta": 1,
    }
    seed = 2026
    np.random.seed(seed)
    x0 = np.random.rand(L)
    z0 = np.random.rand()
    

    cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0, "z0": z0})
    st = time.time()
    rand_seq = cml.generate_rdseq(24)
    rand_seq = rand_seq.T
    print(f"Random sequence generation took {time.time() - st:.4f} seconds")
    if rand_seq.shape != (24, h * w):
        raise ValueError(f"rand_seq shape must be (24, {h * w}), got {rand_seq.shape}")

    encrypted_plane_indices = []

    # 每个通道的高6位分别对应 [0:6]、[8:14]、[16:22]
    for channel in range(3):
        base_idx = channel * 8
        for plane_idx in range(base_idx, base_idx + 6):
            order = np.argsort(rand_seq[plane_idx], axis=None)
            encrypted_bitplanes[plane_idx] = _permute_plane(bitplanes[plane_idx], order)
            decrypted_bitplanes[plane_idx] = _inverse_permute_plane(encrypted_bitplanes[plane_idx], order)
            encrypted_plane_indices.append(plane_idx)

    encrypted_img = _bitplanes_to_image(encrypted_bitplanes)
    decrypted_img = _bitplanes_to_image(decrypted_bitplanes)

    decrypt_ok = np.array_equal(img_arr, decrypted_img)
    changed_mask = np.any(img_arr != encrypted_img, axis=2)
    changed_ratio = changed_mask.mean()

    print(f"image path: {img_pth}")
    print(f"image size: {w}x{h}")
    print(f"encrypted bit planes: {encrypted_plane_indices}")
    print(f"pixel change ratio after encryption: {changed_ratio:.4%}")
    print(f"decryption exact recovery: {decrypt_ok}")

    _show_demo_results(img_arr, encrypted_img, decrypted_img, changed_mask)





if __name__ == "__main__":
    demo_1()
