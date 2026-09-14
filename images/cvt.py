from PIL import Image
import numpy as np

a = np.asarray(Image.open('./images/2.1.06.tiff'))  # 读取 tiff 图像
Image.fromarray(a).save('./images/out.png')   # 像素与 tiff 完全一致