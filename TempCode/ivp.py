import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def lorenz_system(t, state, a, sigma, b):
    """
    定义Lorenz系统的微分方程
    state = [X, Y, Z]
    """
    X, Y, Z = state
    dXdt = a * (Y - X)
    dYdt = (sigma - Z) * X - Y
    dZdt = X * Y - b * Z
    return [dXdt, dYdt, dZdt]

# 设置参数（图片中提到的混沌参数）
a = 10
sigma = 28  # 注意：图片中b=28对应这里的sigma
b = 8/3     # 图片中的σ=8/3对应这里的b

# 初始条件
X0, Y0, Z0 = 1.0, 1.0, 1.0
initial_state = [X0, Y0, Z0]

# 时间范围
t_span = (0, 50)  # 从0到50秒
t_eval = np.linspace(0, 50, 5000)  # 时间点

# 求解微分方程
solution = solve_ivp(lorenz_system, t_span, initial_state,
                    args=(a, sigma, b), t_eval=t_eval, method='RK45')

# 提取结果
X = solution.y[0]
Y = solution.y[1]
Z = solution.y[2]
t = solution.t

# 创建3D图显示混沌吸引子
fig = plt.figure(figsize=(15, 5))

# 3D轨迹图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(X, Y, Z, lw=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Lorenz吸引子')

# 三个变量的时间序列
ax2 = fig.add_subplot(132)
ax2.plot(t, X, label='X')
ax2.plot(t, Y, label='Y')
ax2.plot(t, Z, label='Z')
ax2.set_xlabel('时间')
ax2.set_ylabel('变量值')
ax2.legend()
ax2.set_title('时间序列')
ax2.grid(True)

# 二维投影（X-Y平面）
ax3 = fig.add_subplot(133)
ax3.plot(X, Y, lw=0.5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('X-Y平面投影')
ax3.grid(True)

plt.tight_layout()
plt.show()