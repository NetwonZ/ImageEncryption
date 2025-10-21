import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def normalize_image(imgs:list):
    """将imgs所有图片转化为灰度图，并将大小统一为256*256
    Args:
        imgs (list): 图片列表
    Returns:imgs (list): 处理后的图片列表
    """
    normalized_imgs = []
    for img in imgs:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (256, 256))
        normalized_imgs.append(resized)
    return normalized_imgs

def combine_images(imgs:list,layout=(1,3)):
    """将imgs所有图片按layout的行列数进行拼接
    Args:
        imgs (list): 图片列表
        layout (tuple, optional): 拼接的行列数. Defaults to (2,2).
    Returns:combined_img (ndarray): 拼接后的图片
    """
    rows, cols = layout
    if len(imgs) != rows * cols:
        raise ValueError("Number of images does not match layout")


    h, w = imgs[0].shape[:2]

    combined_img = np.zeros((h * rows, w * cols), dtype=imgs[0].dtype)

    for i in range(rows):
        for j in range(cols):
            combined_img[i*h:(i+1)*h, j*w:(j+1)*w] = imgs[i*cols + j]

    return combined_img

def generate_sk(big_imgs:np.ndarray):
    """
    根据大图生成秘钥种子Kz
    Args:
        big_imgs (ndarray): 大图
    Returns:sk (ndarray): 秘钥
    """
    max_z = 10
    h, w = big_imgs.shape
    kz = np.zeros((max_z,), dtype=np.float32)
    #如果h不能被max_z整除，则将图片高度补齐到能被max_z整除
    if h % max_z != 0:
        new_h = (h // max_z + 1) * max_z
        padded_img = np.zeros((new_h, w), dtype=big_imgs.dtype)
        padded_img[:h, :] = big_imgs
        big_imgs = padded_img
        h = new_h
    for tz in range(1, max_z + 1):#TODO 在论文公式中 ti是(z-1)*h/8+1到z*h/8,但是这显然会导致ti的值不是整数并且会超出图片的行数
        #生成一个ti序列 其内容为(tz-1)*h/10+1到tz*h/10
        ti = np.arange((tz - 1) * h // max_z, tz * h // max_z)
        #取出big_imgs的ti行
        img_ti = big_imgs[ti, :]
        sum_img_ti = np.sum(img_ti)
        kz[tz - 1] = (sum_img_ti % 256)/256
    sk = np.zeros((10,), dtype=np.float32)
    sk[0] = kz[0]
    for i in range(1, 10):
        sk[i] = (kz[i] + sk[i - 1])/2
    return sk


def shuffle_img(img:np.ndarray,x10,r10,x20,r20):
    """
    使用混沌映射，并且初始参数为x10,r10对big_imgs的行进行打乱
                初始参数为x20,r20对big_imgs的列进行打乱
    Args:
        img (ndarray): 待置乱的图像
        x10 (float):
        r10 (float):
        x20 (float):
        r20 (float):
    """
    h,w = img.shape
    col_indices = np.arange(w)
    row_indices = np.arange(h)
    row_seq = logistic_map(x10, r10, h)
    col_seq = logistic_map(x20, r20, w)
    row_sorted_indices = np.argsort(row_seq)
    col_sorted_indices = np.argsort(col_seq)
    shuffled_img = img[row_sorted_indices, :][:, col_sorted_indices]
    return shuffled_img


def logistic_map(x, r, n):
    """生成Logistic映射序列
    Args:
        x (float): 初始值
        r (float): 控制参数
        n (int): 序列长度
    Returns:
        list: Logistic映射序列
    """
    seq = np.zeros(n)
    seq[0] = x
    for i in range(1, n):
        seq[i] = r * seq[i - 1] * (1 - seq[i - 1])
    return seq

def divide2blocks(img:np.ndarray, block_size:tuple):
    """将大图img按block_size划分为多个小块
    Args:
        img (ndarray): 大图
        block_size (tuple): 小块大小
    Returns:
        list: 小块列表
    """
    h, w = img.shape
    bh, bw = block_size[0], block_size[1]
    #如何h，w不能被bh，bw整除，则将图片补齐到能被bh，bw整除
    if h % bh != 0:
        new_h = (h // bh + 1) * bh
        padded_img = np.zeros((new_h, w), dtype=img.dtype)
        padded_img[:h, :] = img
        img = padded_img
        h = new_h
    if w % bw != 0:
        new_w = (w // bw + 1) * bw
        padded_img = np.zeros((h, new_w), dtype=img.dtype)
        padded_img[:, :w] = img
        img = padded_img
        w = new_w

    num_blocks_w = w // bw
    blocks = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = img[i:i+bh, j:j+bw]
            blocks.append(block)
    return blocks,num_blocks_w

def combin2bigimg(PIES:list,number_block_w:int):
    """将小块列表PIES按number_block_w的列数进行拼接为大图
    Args:
        PIES (list): 小块列表
        number_block_w (int): 每行小块数
    Returns:
        ndarray: 拼接后的大图
    """
    if len(PIES) % number_block_w != 0:
        raise ValueError("块数无法整除，不行完整拼接")

    bh, bw = PIES[0].shape
    num_blocks_h = len(PIES) // number_block_w
    h = num_blocks_h * bh
    w = number_block_w * bw
    big_img = np.zeros((h, w), dtype=np.float32)
    for i in range(len(PIES)):
        row = i // number_block_w
        col = i % number_block_w
        big_img[row*bh:(row+1)*bh, col*bw:(col+1)*bw] = PIES[i]

    return big_img
