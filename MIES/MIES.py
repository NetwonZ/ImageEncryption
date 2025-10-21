
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from function import *


path = "../Testimg/"
imgs = load_images_from_folder(path)
# sk = generate_sk(big_img)
sk = [0.1,0.2,0.3,0.4,3.9985,3.9988,3.9984,3.9986,0.001,0.002,3,1,1,5.9]
divided_size = (16,16)
x10,x20,x30,x40,r10,r20,r30,r40,alpha,beta,a,b,c,k = sk
imgs = normalize_image(imgs)

big_img = combine_images(imgs, layout=(1,3))


PIES,number_block_w = divide2blocks(big_img, divided_size)
for i in range(len(PIES)):
    PIES[i] = shuffle_img(PIES[i],x10,r10,x20,r20)
big_img_shuffled = combin2bigimg(PIES,number_block_w)

#生成两个混沌序列
seq_a,seq_b = two_dim_ecomap(5000,a, b, c, k, alpha, beta)
#对混沌序列进行预处理
a_proc = np.floor(seq_a[-1]*1e14 %256)
b_proc = np.floor(seq_b[-2]*1e14 %256)

big_img_his = Histogram(big_img)
big_img_shuffled_his = Histogram(big_img_shuffled)
plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(big_img,cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.title("Original Image Histogram")
plt.bar(range(256),big_img_his,color='black')
plt.xlim([0,255])
plt.subplot(2,2,3)
plt.title("PIES shuffle Histogram")
plt.bar(range(256),big_img_shuffled_his,color='black')
plt.xlim([0,255])
plt.subplot(2,2,4)
plt.title("PIES shuffle")
plt.imshow(big_img_shuffled,cmap='gray')
plt.axis('off')
plt.show()


