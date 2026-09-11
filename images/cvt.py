from PIL import Image
import numpy as np

a = np.asarray(Image.open('./images/4.2.03.tiff'))
Image.fromarray(a).save('./images/out.png')   # 像素与 tiff 完全一致