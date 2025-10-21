def Twodim_logistic_map(x, y, a, b):
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


def generate_sha256_sequence(input_string):
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
