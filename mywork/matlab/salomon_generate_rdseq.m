function Matrix = salomon_generate_rdseq(x0, mu, v, alpha, beta, xi, eta, N, is_mod)
%SALOMON_GENERATE_RDSEQ Salomon 耦合映射格 (CML) 迭代, 生成 (N, L) 随机矩阵。
% 对应 Python 端 mywork/SalomonCouplingCML.py 的 generate_rdseq_fast()。
%
% 系统:
%   f(x)  = |sin((5+3*mu)*(1 - v*x*sin(15*pi*x*(1-x))))|
%   x_{n+1}(i) = mod( (10^alpha - cos(2*pi*(f_{i-1}+f_i+f_{i+1}))) mod 1
%                    + (10^beta * sqrt(f_p^2 + f_q^2)) mod 1, 1 )
%   p = mod((1+xi)*i, L),  q = mod((eta+xi*eta+1)*i, L),  i = 0..L-1
%
% 输入:
%   x0     L×1 初始状态 (来自 salomon_keystream 的 X)
%   mu, v, alpha, beta  控制参数 (double)
%   xi, eta  耦合下标参数 (整数)
%   N      迭代次数
%   is_mod 是否取模到 [0,1) (默认 true)
% 输出:
%   Matrix  (N, L) double
%
% 注: 与 Python numba 路径公式一致; 浮点微差 (sin/cos 1 ULP) 不影响流程一致性。

    if nargin < 9 || isempty(is_mod)
        is_mod = true;
    end
    x0 = double(x0(:));
    L = numel(x0);

    % p/q 下标 (0-based -> 1-based)
    i0 = (0:L - 1).';
    p_idx = mod((1 + xi) * i0, L) + 1;
    q_idx = mod((eta + xi * eta + 1) * i0, L) + 1;

    factor = 5.0 + 3.0 * mu;
    alpha_scale = 10.0 ^ alpha;
    beta_scale = 10.0 ^ beta;
    angle_factor = 2.0 * pi;

    Matrix = zeros(N, L);
    x = x0;
    for t = 1:N
        fx = abs(sin(factor * (1.0 - (v .* x .* sin(15.0 * pi .* x .* (1.0 - x))))));
        fx_left = [fx(end); fx(1:end - 1)];
        fx_right = [fx(2:end); fx(1)];
        fx_p = fx(p_idx);
        fx_q = fx(q_idx);

        alpha_part = alpha_scale - cos(angle_factor * (fx_left + fx + fx_right));
        beta_part = beta_scale .* sqrt(fx_p .* fx_p + fx_q .* fx_q);
        if is_mod
            x_next = mod(mod(alpha_part, 1.0) + mod(beta_part, 1.0), 1.0);
        else
            x_next = alpha_part + beta_part;
        end
        Matrix(t, :) = x_next.';
        x = x_next;
    end
end
