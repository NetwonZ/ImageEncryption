import numpy as np
import matplotlib.pyplot as plt


def two_dim_ecomap(n_iter, a=3, b=1, c=1, k=5.9,
                   alpha0=0.001, beta0=0.002, gamma_type='sum'):
    """二维混沌经济映射实现"""
    alpha = np.zeros(n_iter)
    beta = np.zeros(n_iter)

    alpha[0] = alpha0
    beta[0] = beta0

    def calc_gamma(a_val, b_val, g_type):
        if g_type == 'sum':
            return a_val + b_val
        else:
            return a_val + b_val  # 默认使用和

    for n in range(n_iter - 1):
        gamma_n = calc_gamma(alpha[n], beta[n], gamma_type)
        if gamma_n <= 0:
            gamma_n = 1e-10

        alpha[n + 1] = alpha[n] + k * (a - c - (b * alpha[n]) / gamma_n - b * np.log(gamma_n))
        beta[n + 1] = beta[n] + k * (a - c - (b * beta[n]) / gamma_n - b * np.log(gamma_n))

    return alpha, beta


def bifurcation_diagram(param_range, param_name='k', n_iter=1000, discard=800, variable='alpha'):
    """
    绘制分叉图

    参数:
    param_range: 参数范围
    param_name: 参数名称
    n_iter: 总迭代次数
    discard: 丢弃的瞬态次数
    variable: 观察的变量('alpha'或'beta')
    """
    plt.figure(figsize=(12, 8))

    for param_value in param_range:
        # 设置参数
        params = {'a': 3, 'b': 1, 'c': 1, 'k': 5.9, 'alpha0': 0.001, 'beta0': 0.002}
        params[param_name] = param_value

        # 迭代系统
        alpha, beta = two_dim_ecomap(n_iter=n_iter, **params)

        # 丢弃瞬态，只保留稳定状态
        if variable == 'alpha':
            stable_values = alpha[discard:]
        else:
            stable_values = beta[discard:]

        # 绘制分叉点
        plt.plot([param_value] * len(stable_values), np.abs(alpha[discard:]-beta[discard:]), '.k', alpha=1, markersize=1)

    plt.xlabel(f'Parameter {param_name}')
    plt.ylabel(f'Stable values of {variable}')
    plt.title(f'Bifurcation Diagram of 2D Economic Map ({param_name} parameter)')
    #限制y轴范围
    plt.ylim(-2,20)
    plt.grid(alpha=0.3)
    plt.show()


# 生成分叉图
k_range = np.linspace(5, 6, 500)  # k参数范围
bifurcation_diagram(k_range, param_name='k', variable='alpha')