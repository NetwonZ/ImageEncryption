from pathlib import Path
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil_image

# 保持你原有的类型别名
ImageInput = Union[str, Path, np.ndarray]

# ==========================================
# 1. 保持你原有的底层图像加载和转换函数
# ==========================================
def _load_image_array(image: ImageInput) -> np.ndarray:
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


# ==========================================
# 2. 新增：实现论文图 (a) 的 3D 图像堆叠效果
# ==========================================
def plot_3d_image_stack(images: List[ImageInput], title: str = "3D Image Stack", figsize: tuple = (10, 7)):
    """
    将多张图片像卡片一样，沿着 Y 轴立体堆叠显示。
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    
    for i, img_input in enumerate(images):
        arr = _load_image_array(img_input)
        # 统一转为 RGB 方便贴图
        if arr.ndim == 2:
            img_rgb = np.stack([arr] * 3, axis=-1)
        else:
            img_rgb = arr[:, :, :3]
            
        h, w, _ = img_rgb.shape
        
        # 关键点 1：为防止大图导致 3D 渲染极度卡顿，对贴图进行降采样（控制在 200 像素左右）
        scale = max(1, h // 200, w // 200)
        img_ds = img_rgb[::scale, ::scale]
        h_ds, w_ds, _ = img_ds.shape
        
        # 关键点 2：构建 3D 空间网格。图像立在 X-Z 平面上，Y 轴作为深度
        x = np.linspace(0, w, w_ds)
        z = np.linspace(0, h, h_ds)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, i + 1)  # 每一张图对应一个 Y 轴刻度
        
        # 关键点 3：Matplotlib 3D 贴图颜色需归一化到 [0, 1]，且矩阵首行在上方，3D中坐标向上递增，故需上下翻转
        facecolors = np.flipud(img_ds / 255.0)
        
        # 绘制 3D 表面，使用精确对应的颜色矩阵（切片是为了匹配网格面片数量）
        ax.plot_surface(X, Y, Z, facecolors=facecolors[:-1, :-1], rstride=1, cstride=1, shade=False)

    # 视角与坐标轴微调
    ax.set_title(title, pad=20)
    ax.set_ylim(0, len(images) + 1)
    ax.view_init(elev=20, azim=-60)  # 调整到类似论文的立体倾斜视角
    
    fig.tight_layout()
    plt.show()


# ==========================================
# 3. 新增：实现论文图 (b) 和 (c) 的 3D 瀑布直方图效果
# ==========================================
def plot_3d_waterfall_histogram(images: List[ImageInput], title: str = "3D Waterfall Histogram", figsize: tuple = (11, 7)):
    """
    将多张图片的直方图沿着 Y 轴纵深方向依次排开，形成立体瀑布流。
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    
    # 预设一组类似论文中的丰富配色
    colors = ['#8B2635', '#2E7D32', '#1A237E', '#006064', '#4A148C', '#558B2F']
    bins = np.arange(257)
    x = bins[:-1]  # 0 到 255 的横坐标
    
    for i, img_input in enumerate(images):
        arr = _to_uint8_pixels(_load_image_array(img_input))
        counts, _ = np.histogram(arr.ravel(), bins=bins)
        
        y_pos = i + 1  # 当前直方图在 Y 轴上的深度位置
        
        # 关键点：zdir='y'。这意味着：
        # - 柱子的横向分布在 X 轴（0-255 像素值）
        # - 柱子的高度代表 Z 轴（频数 Frequency）
        # - 整体所在的平面固定在 Y = y_pos 上
        ax.bar(x, counts, zs=y_pos, zdir='y', color=colors[i % len(colors)], alpha=0.8, width=1.3)
        
    ax.set_title(title, pad=20)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Image Index')
    ax.set_zlabel('Frequency')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, len(images) + 1)
    
    # 调整视角以匹配论文
    ax.view_init(elev=25, azim=-65)
    
    fig.tight_layout()
    plt.show()


# ==========================================
# 4. 测试运行示例
# ==========================================
if __name__ == "__main__":
    from scramblediffusion import encrypt_image
    
    # 假设你有多张测试图片
    img_paths = [
        Path(r"C:\ImageEncryptionV2\image\img1.png"),
        Path(r"C:\ImageEncryptionV2\image\img2.png"),
        Path(r"C:\ImageEncryptionV2\image\img3.png"),
    ]
    
    # 生成对应的加密图像
    encrypted_imgs = [encrypt_image(p, verbose=False) for p in img_paths]
    
    # ---- 1. 绘制图 (a) 效果：原始图像 3D 堆叠 ----
    plot_3d_image_stack(img_paths, title="Original Image Stack (a)")
    
    # ---- 2. 绘制图 (b) 效果：原始图像的 3D 瀑布直方图 ----
    plot_3d_waterfall_histogram(img_paths, title="Original Images Histogram (b)")
    
    # ---- 3. 绘制图 (c) 效果：加密图像的 3D 瀑布直方图 ----
    plot_3d_waterfall_histogram(encrypted_imgs, title="Encrypted Images Histogram (c)")