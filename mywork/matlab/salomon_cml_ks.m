function salomon_cml_ks(varargin)
%SALOMON_CML_KS  Salomon coupling CML: Benettin-QR + local LE, KS scan, wireframe.
%
%   salomon_cml_ks()              % 扫描参数平面, 保存带时间戳的 .mat, 并绘图
%   salomon_cml_ks('file.mat')    % 只绘制已保存的扫描结果
%   salomon_cml_ks('Points',[21 21], 'Size',32, ...)
%
% 系统 (与 SalomonCouplingCML.py 一致, 1-based 下标):
%   f(x) = |sin((5+3*mu)*(1 - v*x*sin(15*pi*x*(1-x))))|
%   x_{n+1}(i) = 10^alpha - cos(2*pi*(f_{i-1}+f_i+f_{i+1}))
%                + 10^beta * sqrt(f_p^2 + f_q^2)
%   p = mod((1+xi)*i, L),  q = mod((eta+xi*eta+1)*i, L),  i = 0..L-1
%   is_mod: (alpha_part mod 1 + beta_part mod 1) mod 1
%
% Lyapunov:
%   Benettin-QR : 全维雅可比 QR 谱 (与 Python lyapunov_spectrum 相同)
%   局部 LE     : 沿轨道时间平均 log|J_ii| = log|dF_i/dx_i|
%                 (格点局部膨胀率; KS 密度/宽度定义同 cml_workflow.m)
%
% 绘图风格对齐 cml_workflow.m: mesh FaceColor none / EdgeColor interp,
% 自定义紫-青-黄 colormap, Times New Roman, view(-37.5, 28).

    if nargin >= 1
        a1 = varargin{1};
        if (ischar(a1) || isstring(a1)) && exist(char(a1), 'file')
            plot_parameter_plane(char(a1), 'Show', usejava('desktop'));
            return;
        end
    end

    def = struct();
    def.mu = 5.0; def.v = 20.0; def.alpha = 5.0; def.beta = 5.0;
    def.xi = 1; def.eta = 1; def.is_mod = true;

    ip = inputParser;
    addParameter(ip, 'ParamX', 'mu');
    addParameter(ip, 'RangeX', [0.1, 10.0]);
    addParameter(ip, 'ParamY', 'v');
    addParameter(ip, 'RangeY', [0.1, 10.0]);
    addParameter(ip, 'Points', [21, 21]);
    addParameter(ip, 'Size', 32);
    addParameter(ip, 'Transient', 400);
    addParameter(ip, 'Iterations', 800);
    addParameter(ip, 'Seed', 2026);
    addParameter(ip, 'Base', def);
    parse(ip, varargin{:});
    opt = ip.Results;
    base = opt.Base;
    req = {'mu','v','alpha','beta','xi','eta','is_mod'};
    for k = 1:numel(req)
        if ~isfield(base, req{k}), base.(req{k}) = def.(req{k}); end
    end

    here = fileparts(mfilename('fullpath'));
    stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    output_mat = fullfile(here, 'data', sprintf('salomon_ks_scan_%s.mat', stamp));
    output_fig = fullfile(here, 'figures', sprintf('salomon_ks_scan_%s.png', stamp));
    show_fig = usejava('desktop');

    fprintf('=== 开始扫描 Salomon CML 参数平面 (%s x %s) ===\n', opt.ParamX, opt.ParamY);
    mat_file = scan_parameter_plane(opt.ParamX, opt.RangeX, opt.ParamY, opt.RangeY, ...
        'Points', opt.Points, ...
        'Size', opt.Size, ...
        'Transient', opt.Transient, ...
        'Iterations', opt.Iterations, ...
        'Seed', opt.Seed, ...
        'Base', base, ...
        'OutputFile', output_mat, ...
        'Progress', true);

    fprintf('=== 绘制 KS 熵密度 / 熵宽度线框图 ===\n');
    plot_parameter_plane(mat_file, 'Output', output_fig, 'Show', show_fig);
end

%% ==================== 局部映射 f, f' ====================

function [fx, fp] = salomon_f(x, mu, v)
% f(x) = |sin((5+3*mu)*(1 - v*x*sin(15*pi*x*(1-x))))|
% f' 按实 Abs 求导: sign(sin(theta))*cos(theta)*dtheta/dx, sign(0)=0.
    k = 5.0 + 3.0 * mu;
    w = 15.0 * pi;
    u = x .* (1.0 - x);
    s_in = sin(w * u);
    c_in = cos(w * u);
    g = v .* x .* s_in;
    theta = k * (1.0 - g);
    s_th = sin(theta);
    c_th = cos(theta);
    fx = abs(s_th);

    dg = v .* (s_in + x .* c_in .* w .* (1.0 - 2.0 * x));
    dtheta = -k * dg;
    fp = sign(s_th) .* c_th .* dtheta;
end

function [p_idx, q_idx, left_idx, right_idx] = neighbor_indices(L, xi, eta)
    if xi == 0
        eta = L;
    end
    if eta == 0
        xi = L;
    end
    i0 = (0:L-1).';
    idx = (1:L).';
    p_idx = mod((1 + xi) * i0, L) + 1;
    q_idx = mod((eta + xi * eta + 1) * i0, L) + 1;
    left_idx  = mod(idx - 2, L) + 1;
    right_idx = mod(idx, L) + 1;
end

function x_next = cml_step(x, fx, p_idx, q_idx, left_idx, right_idx, alpha, beta, is_mod)
    alpha_part = (10.0 .^ alpha) - cos(2.0 * pi * (fx(left_idx) + fx + fx(right_idx)));
    beta_part  = (10.0 .^ beta) .* sqrt(fx(p_idx).^2 + fx(q_idx).^2);
    if is_mod
        x_next = mod(mod(alpha_part, 1.0) + mod(beta_part, 1.0), 1.0);
    else
        x_next = alpha_part + beta_part;
    end
end

function [Jii, Z] = jacobian_action(x, fx, fp, Q, p_idx, q_idx, left_idx, right_idx, beta, epsJ)
% 雅可比作用于切空间 Q, 同时给出对角元 J_ii (局部 LE).
% 模运算的不连续点测度为零, 与 Python _jacobian_x 一样忽略 wrap.
    S = fx(left_idx) + fx + fx(right_idx);
    sin_term = 2.0 * pi * sin(2.0 * pi * S);

    fx_p = fx(p_idx);
    fx_q = fx(q_idx);
    fp_p = fp(p_idx);
    fp_q = fp(q_idx);
    R = sqrt(fx_p .* fx_p + fx_q .* fx_q);
    inv_R = 1.0 ./ max(R, epsJ);
    coup = 10.0 .^ beta;
    coup_p = coup .* (fx_p .* fp_p) .* inv_R;
    coup_q = coup .* (fx_q .* fp_q) .* inv_R;

    idx = (1:numel(x)).';
    Jii = sin_term .* fp;
    same_p = (p_idx == idx);
    same_q = (q_idx == idx);
    Jii(same_p) = Jii(same_p) + coup_p(same_p);
    Jii(same_q) = Jii(same_q) + coup_q(same_q);

    if nargout > 1
        aL = sin_term .* fp(left_idx);
        aC = sin_term .* fp;
        aR = sin_term .* fp(right_idx);
        Z = aL .* Q(left_idx, :) + aC .* Q + aR .* Q(right_idx, :) ...
            + coup_p .* Q(p_idx, :) + coup_q .* Q(q_idx, :);
    end
end

%% ==================== Lyapunov: QR + 局部 ====================

function [spec_qr, spec_loc] = lyapunov_both(params, N, transient, iterations, seed, epsJ)
    if nargin < 6 || isempty(epsJ)
        epsJ = 1e-12;
    end
    if nargin >= 5 && ~isempty(seed)
        rng(seed, 'twister');
    end

    x = rand(N, 1);
    [p_idx, q_idx, left_idx, right_idx] = neighbor_indices(N, params.xi, params.eta);
    mu = params.mu; v = params.v;
    alpha = params.alpha; beta = params.beta;
    is_mod = params.is_mod;

    Q = eye(N);
    log_qr  = zeros(N, 1);
    log_loc = zeros(N, 1);
    total = transient + iterations;

    for t = 1:total
        [fx, fp] = salomon_f(x, mu, v);
        [Jii, Z] = jacobian_action(x, fx, fp, Q, p_idx, q_idx, left_idx, right_idx, beta, epsJ);
        [Q, R] = qr(Z);
        if t > transient
            d = abs(diag(R));
            log_qr  = log_qr  + log(max(d, 1e-15));
            log_loc = log_loc + log(max(abs(Jii), 1e-15));
        end
        x = cml_step(x, fx, p_idx, q_idx, left_idx, right_idx, alpha, beta, is_mod);
    end

    spec_qr  = sort(log_qr  / double(iterations), 'descend');
    spec_loc = sort(log_loc / double(iterations), 'descend');
end

function [h, hu] = ks_metrics(lambda, N)
    pos = lambda(lambda > 0.0);
    h  = sum(pos) / double(N);
    hu = numel(pos) / double(N);
end

%% ==================== 参数扫描 ====================

function saved_path = scan_parameter_plane(param_x, range_x, param_y, range_y, varargin)
    p = inputParser;
    addParameter(p, 'Points', [21, 21]);
    addParameter(p, 'Size', 32);
    addParameter(p, 'Transient', 400);
    addParameter(p, 'Iterations', 800);
    addParameter(p, 'Seed', 2026);
    addParameter(p, 'Base', struct());
    addParameter(p, 'OutputFile', fullfile('data', 'salomon_ks_scan.mat'));
    addParameter(p, 'Progress', true);
    parse(p, varargin{:});
    opts = p.Results;

    pts = opts.Points;
    xvals = linspace(range_x(1), range_x(2), pts(1));
    yvals = linspace(range_y(1), range_y(2), pts(2));
    N = opts.Size;

    h_qr  = zeros(pts(2), pts(1));
    hu_qr = zeros(pts(2), pts(1));
    h_loc  = zeros(pts(2), pts(1));
    hu_loc = zeros(pts(2), pts(1));

    tic;
    for row = 1:pts(2)
        for col = 1:pts(1)
            params = opts.Base;
            params.(param_x) = xvals(col);
            params.(param_y) = yvals(row);
            if isfield(params, 'xi'),  params.xi  = round(params.xi);  end
            if isfield(params, 'eta'), params.eta = round(params.eta); end

            grid_seed = opts.Seed + (row - 1) * pts(1) + col;
            [spec_qr, spec_loc] = lyapunov_both(params, N, ...
                opts.Transient, opts.Iterations, grid_seed);

            [h_qr(row, col),  hu_qr(row, col)]  = ks_metrics(spec_qr, N);
            [h_loc(row, col), hu_loc(row, col)] = ks_metrics(spec_loc, N);
        end
        if opts.Progress
            fprintf('已完成扫描: %s 轴第 %2d / %2d 行 (%.1f%%)\n', ...
                param_y, row, pts(2), (row / pts(2)) * 100);
        end
    end
    fprintf('扫描完成，用时: %.2f 秒\n', toc);

    [dir_path, ~, ~] = fileparts(opts.OutputFile);
    if ~isempty(dir_path) && ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end

    x = xvals;
    y = yvals;
    meta = struct( ...
        'param_x', param_x, 'param_y', param_y, ...
        'N', N, 'transient', opts.Transient, 'iterations', opts.Iterations, ...
        'seed', opts.Seed, 'base', opts.Base, ...
        'qr_method', 'Benettin QR of full Jacobian (SalomonCouplingCML.py)', ...
        'local_method', 'time-average of log|J_ii| = log|dF_i/dx_i|');
    save(opts.OutputFile, 'x', 'y', 'h_qr', 'hu_qr', 'h_loc', 'hu_loc', ...
        'param_x', 'param_y', 'meta', '-v7.3');
    saved_path = opts.OutputFile;
    fprintf('已保存: %s\n', saved_path);
end

%% ==================== 线框图绘制 ====================

function plot_parameter_plane(data_file, varargin)
    p = inputParser;
    addParameter(p, 'Output', '');
    addParameter(p, 'Show', true);
    parse(p, varargin{:});
    opts = p.Results;

    data = load(data_file);
    x = data.x(:).';
    y = data.y(:).';
    if isfield(data, 'param_x'), px = char(data.param_x); else, px = 'mu'; end
    if isfield(data, 'param_y'), py = char(data.param_y); else, py = 'v'; end

    [X, Y] = meshgrid(x, y);

    c_nodes = [ ...
         91,  73, 182; ...
         92, 144, 245; ...
         37, 193, 190; ...
        161, 211, 128; ...
        244, 232,  80] / 255;
    custom_cmap = interp1(linspace(0, 1, size(c_nodes, 1)), c_nodes, linspace(0, 1, 256));

    vis = 'on';
    if ~opts.Show
        vis = 'off';
    end
    fig = figure('Color', 'w', 'Position', [80, 80, 1100, 920], 'Visible', vis);

    panels = { ...
        data.h_qr,  '(a) KS entropy density  (Benettin-QR)',  'h';  ...
        data.hu_qr, '(b) KS entropy width  (Benettin-QR)',    'h_u'; ...
        data.h_loc, '(c) KS entropy density  (local LE)',     'h';  ...
        data.hu_loc,'(d) KS entropy width  (local LE)',       'h_u'};

    xlab = param_tex(px);
    ylab = param_tex(py);

    for k = 1:4
        ax = subplot(2, 2, k);
        Z = panels{k, 1};
        m = mesh(X, Y, Z);
        set(m, 'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 0.8);
        colormap(ax, custom_cmap);
        xlabel(xlab, 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'tex');
        ylabel(ylab, 'FontName', 'Times New Roman', 'FontSize', 12, 'Interpreter', 'tex');
        zlabel(['\it ', panels{k, 3}], 'FontName', 'Times New Roman', 'FontSize', 12);
        xlim([min(x), max(x)]);
        ylim([min(y), max(y)]);
        zmin = min(Z(:)); zmax = max(Z(:));
        if ~isfinite(zmin) || ~isfinite(zmax) || abs(zmax - zmin) < 1e-12
            zmax = zmin + 1e-6;
        end
        if k == 2 || k == 4
            zlim([0, max(1.0, zmax)]);
        else
            zlim([min(0, zmin), zmax]);
        end
        view(-37.5, 28); grid on; box on;
        set(ax, 'BoxStyle', 'full', 'GridAlpha', 0.2, ...
            'FontName', 'Times New Roman', 'FontSize', 10);
        title(panels{k, 2}, ...
            'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'normal', ...
            'Units', 'normalized', 'Position', [0.5, -0.18, 0]);
    end

    if isempty(opts.Output)
        [d, n, ~] = fileparts(data_file);
        fig_dir = fullfile(d, '..', 'figures');
        if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
        opts.Output = fullfile(fig_dir, [n, '.png']);
    end
    [dir_path, ~, ~] = fileparts(opts.Output);
    if ~isempty(dir_path) && ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
    try
        exportgraphics(fig, opts.Output, 'Resolution', 300);
    catch
        saveas(fig, opts.Output);
    end
    fprintf('已保存图: %s\n', opts.Output);
    if ~opts.Show
        close(fig);
    end
end

function s = param_tex(name)
    switch lower(strtrim(name))
        case 'mu',    s = '\it \mu';
        case 'v',     s = '\it v';
        case 'alpha', s = '\it \alpha';
        case 'beta',  s = '\it \beta';
        case 'xi',    s = '\it \xi';
        case 'eta',   s = '\it \eta';
        otherwise,    s = ['\it ', name];
    end
end
