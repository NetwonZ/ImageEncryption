function blocks = salomon_adaptive_partition(image, b_max, b_min, threshold)
%SALOMON_ADAPTIVE_PARTITION 自适应分块：递归分割高梯度区域。
% 对应 Python 端 Encrypter.adaptive_partition()。
%
% 输入:
%   image     H×W×3 uint8 RGB
%   b_max     初始块尺寸上限 (默认 64)
%   b_min     最小块尺寸 (默认 16)
%   threshold 梯度阈值；空 [] 时用 1.5*mean(gradient)
% 输出:
%   blocks    struct 数组: row, col, height, width (row/col 为 Python 0-based)
%
% 梯度: gray 方法 |∇| = hypot(dI/dr, dI/dc)，dI/dr 用 np.gradient(axis=0) 即
%       中心差分 i+1 与 i-1, 边界用单边差分。

    if nargin < 2 || isempty(b_max), b_max = 64; end
    if nargin < 3 || isempty(b_min), b_min = 16; end
    image = double(image);
    gray = 0.299 * image(:, :, 1) + 0.587 * image(:, :, 2) + 0.114 * image(:, :, 3);

    [H, W] = size(gray);
    % np.gradient along axis 0 (rows)
    gr = zeros(H, W);
    gr(1, :) = gray(2, :) - gray(1, :);
    gr(H, :) = gray(H, :) - gray(H - 1, :);
    gr(2:H - 1, :) = (gray(3:H, :) - gray(1:H - 2, :)) / 2.0;
    % np.gradient along axis 1 (cols)
    gc = zeros(H, W);
    gc(:, 1) = gray(:, 2) - gray(:, 1);
    gc(:, W) = gray(:, W) - gray(:, W - 1);
    gc(:, 2:W - 1) = (gray(:, 3:W) - gray(:, 1:W - 2)) / 2.0;
    gradient = hypot(gr, gc);

    if isempty(threshold)
        threshold = 1.5 * mean(gradient(:));
    end

    blocks = struct('row', {}, 'col', {}, 'height', {}, 'width', {}, 'mean_gradient', {});

    function split(row, col, bh, bw)
        g = mean(mean(gradient(row + 1:row + bh, col + 1:col + bw)));
        can_split = bh > 1 && bw > 1;
        reached_min = bh <= b_min || bw <= b_min;
        if reached_min || g <= threshold || ~can_split
            blocks(end + 1) = struct('row', row, 'col', col, 'height', bh, 'width', bw, 'mean_gradient', g);
            return;
        end
        hh = floor(bh / 2);
        hw = floor(bw / 2);
        if hh == 0 || hw == 0
            blocks(end + 1) = struct('row', row, 'col', col, 'height', bh, 'width', bw, 'mean_gradient', g);
            return;
        end
        split(row, col, hh, hw);
        split(row, col + hw, hh, bw - hw);
        split(row + hh, col, bh - hh, hw);
        split(row + hh, col + hw, bh - hh, bw - hw);
    end

    for row = 0:b_max:H - 1
        for col = 0:b_max:W - 1
            split(row, col, min(b_max, H - row), min(b_max, W - col));
        end
    end
end
