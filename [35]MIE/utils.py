def Twodim_logistic_map(x, y, r1, r2, s1, s2):
    if r1 is None or r2 is None or s1 is None or s2 is None:
        from constants import Towdim_LOGISTIC_MAP_r1, Towdim_LOGISTIC_MAP_r2, Towdim_LOGISTIC_MAP_s1, Towdim_LOGISTIC_MAP_s2
        r1,r2,s1,s2 = Towdim_LOGISTIC_MAP_r1, Towdim_LOGISTIC_MAP_r2, Towdim_LOGISTIC_MAP_s1, Towdim_LOGISTIC_MAP_s2
    x_next = r1*x*(1-x)+s1*y**2
    y_next = r2*y*(1-y)+s2*(x**2+x*y)
    return x_next, y_next


def Chen_chaotic_system(x, y, z):
    from constants import CHEN_CHAOTIC_SYSTEM_a, CHEN_CHAOTIC_SYSTEM_b, CHEN_CHAOTIC_SYSTEM_c
    a,b,c = CHEN_CHAOTIC_SYSTEM_a, CHEN_CHAOTIC_SYSTEM_b, CHEN_CHAOTIC_SYSTEM_c
    dxdt = a*(y - x)
    dydt = (c - a)*x - x*z + c*y
    dzdt = x*y - b*z
    return dxdt, dydt, dzdt


def Generate_sha256_sequence(input_string):
    """
    return: 长度为32的整数列表,每个整数在0-255之间
    """
    import hashlib
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    bin_dig = bin(int(hex_dig, 16))[2:].zfill(256)
    #每八位转化为一个整数，存入seq
    seq = []
    for i in range(0, 256, 8):
        byte = bin_dig[i:i+8]
        seq.append(int(byte, 2))
    return seq

def Show_TDLM():
    import matplotlib.pyplot as plt
    n_iterations = 10000
    n_transient = 1000
    x,y = 1,1
    x_list = []
    y_list = []
    for _ in range(n_iterations):
        x,y = Twodim_logistic_map(x, y, 0, 0)
        x_list.append(x)
        y_list.append(y)

    plt.subplot(3,1,1)
    plt.plot(x_list[n_transient:], y_list[n_transient:], '.', markersize=1)
    plt.title('Two-dimensional Logistic Map')
    plt.xlabel('x')
    plt.ylabel('y')
    #绘制x和y随迭代次数变化的图像

    plt.subplot(3,1,2)
    plt.plot(range(n_iterations), x_list, '-', markersize=1)
    plt.title('x over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.subplot(3,1,3)
    plt.plot(range(n_iterations), y_list, '-', markersize=1)
    plt.title('y over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('y value')

    plt.show()

def Show_Chen():
    import matplotlib.pyplot as plt
    n_iterations = 50000
    dt = 0.001
    x,y,z = 1,1,1
    x_list = []
    y_list = []
    z_list = []
    #使用四阶龙格库塔法进行迭代
    for _ in range(n_iterations):
        k1x, k1y, k1z = Chen_chaotic_system(x, y, z)
        k2x, k2y, k2z = Chen_chaotic_system(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z)
        k3x, k3y, k3z = Chen_chaotic_system(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z)
        k4x, k4y, k4z = Chen_chaotic_system(x + dt*k3x, y + dt*k3y, z + dt*k3z)

        x += (dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
        y += (dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
        z += (dt/6)*(k1z + 2*k2z + 2*k3z + k4z)

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_list, y_list, z_list, lw=0.5)
    ax.set_title('Chen Chaotic System Attractor')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def Generate_KeyStream(HashSeq: list[int]) -> list[float]:
    key_stream = []
    x1 = (HashSeq[0] ^ HashSeq[1])/255
    y1 = (HashSeq[2] ^ HashSeq[3])/255
    r1 = 2.75+(HashSeq[4] ^ HashSeq[5]^HashSeq[6]^HashSeq[7])/255*0.65
    r2 = 2.75+(HashSeq[8] ^ HashSeq[9]^HashSeq[10]^HashSeq[11])/255*0.7
    s1 = 0.15+(HashSeq[12] ^ HashSeq[13]^HashSeq[14]^HashSeq[15])/255*0.06
    s2 = 0.13+(HashSeq[16] ^ HashSeq[17]^HashSeq[18]^HashSeq[19])/255*0.02
    x0 = (HashSeq[20] ^ HashSeq[21]^HashSeq[22]^HashSeq[23])/255
    y0 = (HashSeq[24] ^ HashSeq[25]^HashSeq[26]^HashSeq[27])/255
    z0 = (HashSeq[28] ^ HashSeq[29]^HashSeq[30]^HashSeq[31])/255
    return [x1, y1, r1, r2, s1, s2, x0, y0, z0]

def Img2Bits(img):
    import numpy as np
    h, w = img.shape
    bits = np.unpackbits(img.reshape(-1,1), axis=1)
    return bits.reshape(h, w, 8)

def Scrambling_image(bits_imgs,T,dim):
    """
    使用T对第dim维度进行像素位置打乱
    """
    import numpy as np

    for img in bits_imgs:
        h, w, c = img.shape
        targetdim_flatten = img[:,:,dim-1].reshape(-1)
        scrambled_flatten = targetdim_flatten[T]
        img[:,:,dim-1] = scrambled_flatten.reshape(h, w)

def Bits2Img(bits_img):
    import numpy as np
    h, w, c = bits_img.shape
    img = np.packbits(bits_img.reshape(-1,8), axis=1)
    return img.reshape(h, w)

def Show_imgs(imgs,mode):
    if mode=='bits':
        i = 1
        for bits_img in imgs:
            import matplotlib.pyplot as plt
            img = Bits2Img(bits_img)
            plt.subplot(2,4,i)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            i += 1
        plt.show()
    elif mode=='img':
        import matplotlib.pyplot as plt
        i = 1
        for img in imgs:
            plt.subplot(2,4,i)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            i += 1
        plt.show()

def Scrambling_BitPlane(bits_imgs,K):
    import numpy as np
    #生成一个0-8*K-1的序列,并且随机打乱
    Q = np.arange(0, 8*K)
    np.random.shuffle(Q)
    Bit_combine_imgs = np.concatenate(bits_imgs, axis=2)  

    #对每一位平面进行打乱
    Bit_combine_imgs_scrambled = np.empty_like(Bit_combine_imgs)
    for i in range(8*K):
        Bit_combine_imgs_scrambled[:,:,i] = Bit_combine_imgs[:,:,Q[i]]
    #将打乱后的位平面重新分割为K个图像
    scrambled_bits_imgs = []
    for i in range(K):
        scrambled_bits_imgs.append(Bit_combine_imgs_scrambled[:,:,i*8:(i+1)*8])
    return scrambled_bits_imgs  