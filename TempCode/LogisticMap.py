import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# 根据图片中的公式定义洛伦兹系统
def lorenz_system(t, state, sigma, a, b):
    """
    定义洛伦兹系统的微分方程
    根据图片中的公式：
    dX/dt = a(Y - X)
    dY/dt = (σ - Z)X - Y
    dZ/dt = XY - bZ
    """
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (sigma - z) * x - y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]


sigma = 28
a = 10
b = 8/3

# 初始条件
initial_state = [1, 1, 1]  # [X0, Y0, Z0]

# 时间范围
t_span = (0, 500)  # 从0到50秒
t_eval = np.linspace(0, 200, 10000)  # 时间点

# 使用solve_ivp求解微分方程（内部使用龙格-库塔法）
solution = solve_ivp(lorenz_system, t_span, initial_state,
                     args=(sigma, a, b), t_eval=t_eval, method='RK45')

# 提取结果
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# 创建3D图形
fig = plt.figure(figsize=(15, 5))

# 1. 3D吸引子图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(x, y, z, lw=0.5, color='blue', alpha=0.7)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Lorenz Attractor')

# 2. XY平面投影
ax2 = fig.add_subplot(132)
ax2.plot(x, y, lw=0.5, color='red', alpha=0.7)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('XY Plane Projection')
ax2.grid(True, alpha=0.3)

# 3. XZ平面投影
ax3 = fig.add_subplot(133)
ax3.plot(x, z, lw=0.5, color='green', alpha=0.7)
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('XZ Plane Projection')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 额外：时间序列图
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_eval, x, 'b-', lw=1, label='X(t)')
plt.ylabel('X')
plt.title('Lorenz System Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t_eval, y, 'r-', lw=1, label='Y(t)')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t_eval, z, 'g-', lw=1, label='Z(t)')
plt.xlabel('Time')
plt.ylabel('Z')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印系统信息
print("洛伦兹系统参数（根据图片）:")
print(f"σ = {sigma}")
print(f"a = {a}")
print(f"b = {b}")
print(f"初始条件: X0={initial_state[0]}, Y0={initial_state[1]}, Z0={initial_state[2]}")