import cv2
from utils import *
import numpy as np
import os


def just_testmain():
    imgs = []
    M, N = 512, 512
    
    for i in os.listdir("D:\ImageEncryption\Testimg"):
        t_img = cv2.imread(os.path.join("D:\ImageEncryption\Testimg", i), cv2.IMREAD_GRAYSCALE)
        imgs.append(t_img)
    print("Loaded", len(imgs), "images for testing.")
    K = len(imgs)
    #将图像转化为位平面
    bits_imgs = [Img2Bits(img) for img in imgs]
    #使用SHA-256生成密钥流
    hash_seq = Generate_sha256_sequence("Hello, World!")
    x1, y1, r1, r2, s1, s2, x0, y0, z0 = Generate_KeyStream(hash_seq)
    
    #使用x1,y1,r1,r2,s1,s2作为参数，进行二维Logistic映射迭代，两个M*N的混沌序列-X1和X2
    X1, X2 = [], []
    for i in range(M*N+3000):
        x1, y1 = Twodim_logistic_map(x1, y1, r1, r2, s1, s2)
        if i >= 3000:
            X1.append(x1)
            X2.append(y1)
    #使用x0,y0,z0作为初值，使用四阶龙格库塔对Chen混沌系统迭代，生成三个M*N的混沌序列-X3,X4,X5
    X3, X4, X5 = [], [], []
    dt = 0.001
    for i in range(M*N+3000):
        k1x, k1y, k1z = Chen_chaotic_system(x0, y0, z0)
        k2x, k2y, k2z = Chen_chaotic_system(x0 + 0.5*dt*k1x, y0 + 0.5*dt*k1y, z0 + 0.5*dt*k1z)
        k3x, k3y, k3z = Chen_chaotic_system(x0 + 0.5*dt*k2x, y0 + 0.5*dt*k2y, z0 + 0.5*dt*k2z)
        k4x, k4y, k4z = Chen_chaotic_system(x0 + dt*k3x, y0 + dt*k3y, z0 + dt*k3z)
        x0 += (dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
        y0 += (dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
        z0 += (dt/6)*(k1z + 2*k2z + 2*k3z + k4z)
        if i >= 3000:
            X3.append(x0)
            X4.append(y0)
            X5.append(z0)

    Show_imgs(bits_imgs,mode='bits')
    #对X1,X2,X3,X4,X5进行升序排序，得到排序索引序列，分别记为I1,I2,I3,I4,I5
    I1 = np.argsort(X1)
    I2 = np.argsort(X2)
    I3 = np.argsort(X3)
    I4 = np.argsort(X4)
    I5 = np.argsort(X5)
    Scrambling_image(bits_imgs, I1, 5)
    Scrambling_image(bits_imgs, I2, 4)
    Scrambling_image(bits_imgs, I3, 3)
    Scrambling_image(bits_imgs, I4, 2)
    Scrambling_image(bits_imgs, I5, 1)
    
    Show_imgs(bits_imgs,mode='bits')

    scrambled_bits_imgs = Scrambling_BitPlane(bits_imgs,K)
    
    Show_imgs(scrambled_bits_imgs,mode='bits')

if __name__ == "__main__":
    just_testmain()
