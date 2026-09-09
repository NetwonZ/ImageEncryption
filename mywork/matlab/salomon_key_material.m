function key = salomon_key_material(cml, blocks, height, width, seq_rows)
%SALOMON_KEY_MATERIAL 从混沌矩阵生成全部密钥材料。
% 对应 Python 端 Encrypter.generate_key_material() (global_bit_permutation_v2 路径)。
%
% 输入:
%   cml       struct: X (L×1 初始状态), mu, v, alpha, beta, xi, eta
%   blocks    salomon_adaptive_partition 输出
%   height,width  图像尺寸
%   seq_rows  可选: generate_rdseq 的额外行数 (默认按 Python 公式)
% 输出:
%   key   struct:
%     EncodeRule, shuffled_rule_ids (B×1 uint8)
%     intra_permutations (B×1 cell of struct{plane,row,col} int32, 1-based)
%     shift_a (H×W int32), shift_b (H×12 int32), shift_c (W×12 int32)
%     block_operation_rule_ids (B×1 uint8)
%     global_permutation (1×L int32, 1-based)
%     global_key_matrix (12×H×W uint8)
%     global_initial_vector (12×1 uint8)
%     block_key_matrices (B×1 cell of 12×h×w uint8)
%     global_key_rule_id (1×L uint8)
%
% 所有像素线性索引均按 Python C-order: p = (i-1)*W + j。

    L = height * width;
    B = numel(blocks);
    DNA_PLANE_COUNT = 12;

    % ---- 需要的混沌值个数 (与 Python 一致) ----
    N_InnerPermutation = 0;
    for b = 1:B
        N_InnerPermutation = N_InnerPermutation + DNA_PLANE_COUNT + blocks(b).height + blocks(b).width;
    end
    N_GlobalBitPermutationV2 = height * width + DNA_PLANE_COUNT * (height + width);
    N_GlobalPermutation = height * width;
    N_GlobalKeyRule = height * width;
    required_values = B + N_InnerPermutation + N_GlobalBitPermutationV2 + B + ...
        N_GlobalPermutation + N_GlobalKeyRule;
    pixel_count = L;
    if nargin < 5 || isempty(seq_rows)
        % Python: (required_values + pixel_count - 1) // pixel_count.
        % This is already ceiling division; do not apply ceil a second time.
        sequence_row_count = max(3, floor((required_values + pixel_count - 1) / pixel_count));
    else
        sequence_row_count = seq_rows;
    end

    Matrix = salomon_generate_rdseq(cml.X, cml.mu, cml.v, cml.alpha, cml.beta, ...
        cml.xi, cml.eta, 4 + sequence_row_count, true);

    % ---- 序列构造: 从 Matrix 的最后一行往前取 sequence_row_count 行 ----
    Seq = zeros(sequence_row_count * pixel_count, 1);
    for r = 1:sequence_row_count
        row = Matrix(4 + sequence_row_count - r + 1, :);
        Seq((r - 1) * pixel_count + 1:r * pixel_count) = row;
    end
    if required_values > numel(Seq)
        error('Seq length insufficient for key allocations');
    end

    % Python allocates EncodeRule first, then advances the sequence cursor.
    cursor = B;
    EncodeRule = mod(floor(Seq(1:B) * 10^10), 8) + 1;

    % ---- 块内三维轴置换 (intra_permutations) ----
    intra_permutations = cell(B, 1);
    for b = 1:B
        blk = blocks(b);
        pl = Seq(cursor + 1:cursor + DNA_PLANE_COUNT); cursor = cursor + DNA_PLANE_COUNT;
        rw = Seq(cursor + 1:cursor + blk.height); cursor = cursor + blk.height;
        cl = Seq(cursor + 1:cursor + blk.width); cursor = cursor + blk.width;
        [~, plane_p] = sort(pl);   % 与 np.argsort 一致 (默认升序)
        [~, row_p] = sort(rw);
        [~, col_p] = sort(cl);
        intra_permutations{b} = struct('plane', int32(plane_p), 'row', int32(row_p), 'col', int32(col_p));
    end

    % ---- A/B/C 矩阵 (C-order 显式映射 p=(i-1)*W+j) ----
    matrix_a = reshape(Seq(cursor + 1:cursor + pixel_count), width, height).'; % 转置得 H×W (C-order)
    cursor = cursor + pixel_count;
    matrix_b = reshape(Seq(cursor + 1:cursor + DNA_PLANE_COUNT * height), DNA_PLANE_COUNT, height).';
    cursor = cursor + DNA_PLANE_COUNT * height;
    matrix_c = reshape(Seq(cursor + 1:cursor + DNA_PLANE_COUNT * width), DNA_PLANE_COUNT, width).';
    cursor = cursor + DNA_PLANE_COUNT * width;
    shift_a = int32(mod(floor(matrix_a * 10^14), DNA_PLANE_COUNT));
    shift_b = int32(mod(floor(matrix_b * 10^14), width));
    shift_c = int32(mod(floor(matrix_c * 10^14), height));

    % ---- 块 DNA 运算规则 ----
    opvals = Seq(cursor + 1:cursor + B); cursor = cursor + B;
    block_operation_rule_ids = uint8(mod(floor(opvals * 10^10), 3) + 1);

    % ---- 全局置换与全局密钥规则 ----
    gvals = Seq(cursor + 1:cursor + N_GlobalPermutation); cursor = cursor + N_GlobalPermutation;
    [~, global_permutation] = sort(gvals);
    gkvals = Seq(cursor + 1:cursor + N_GlobalKeyRule); cursor = cursor + N_GlobalKeyRule;
    global_key_rule_id = uint8(mod(floor(gkvals * 10^10), 8) + 1);
    if cursor ~= required_values
        error('key sequence allocation mismatch');
    end

    % ---- 像素 -> 块规则映射 (C-order) ----
    rule_map = zeros(height, width, 'uint8');
    for b = 1:B
        blk = blocks(b);
        rule_map(blk.row + 1:blk.row + blk.height, blk.col + 1:blk.col + blk.width) = uint8(EncodeRule(b) - 1);
    end

    % ---- 块密钥矩阵: Matrix 第 1、2 行 (Python 0,1) ----
    % Python: DNA_RULE_TABLE[rule_map(p), pair_values(plane,p)] (C-order)
    T = double(dna_rule_table());
    block_key_values = sequence_pair_to_dna_values(Matrix(1, :), Matrix(2, :), pixel_count);  % 12×L
    full_block_key = zeros(DNA_PLANE_COUNT, height, width, 'uint8');
    for p = 1:pixel_count
        i = floor((p - 1) / width) + 1;
        j = mod(p - 1, width) + 1;
        full_block_key(:, i, j) = uint8(T(rule_map(i, j) + 1, double(block_key_values(:, p)) + 1));
    end
    block_key_matrices = cell(B, 1);
    for b = 1:B
        blk = blocks(b);
        block_key_matrices{b} = full_block_key(:, blk.row + 1:blk.row + blk.height, blk.col + 1:blk.col + blk.width);
    end

    % ---- 全局密钥矩阵: Matrix 第 3、4 行 (Python 2,3) ----
    global_key_values = sequence_pair_to_dna_values(Matrix(3, :), Matrix(4, :), pixel_count);  % 12×L
    global_key_matrix = zeros(DNA_PLANE_COUNT, height, width, 'uint8');
    for p = 1:pixel_count
        i = floor((p - 1) / width) + 1;
        j = mod(p - 1, width) + 1;
        global_key_matrix(:, i, j) = uint8(T(double(global_key_rule_id(p)), double(global_key_values(:, p)) + 1));
    end

    key = struct();
    key.EncodeRule = uint8(EncodeRule);
    key.shuffled_rule_ids = uint8(EncodeRule);   % v2 路径不交换规则
    key.intra_permutations = intra_permutations;
    key.shift_a = shift_a;
    key.shift_b = shift_b;
    key.shift_c = shift_c;
    key.block_operation_rule_ids = block_operation_rule_ids;
    key.global_permutation = int32(global_permutation);
    key.global_key_matrix = global_key_matrix;
    key.global_initial_vector = zeros(DNA_PLANE_COUNT, 1, 'uint8');
    key.block_key_matrices = block_key_matrices;
    key.global_key_rule_id = global_key_rule_id;
end

function pair_values = sequence_pair_to_dna_values(low_seq, high_seq, pixel_count)
%SEQUENCE_PAIR_TO_DNA_VALUES 两条混沌序列 -> (12, pixel_count) 两比特值
%   pair_values(plane, p) = 2*bit(11-k, high_q(p)) + bit(11-k, low_q(p)),  plane=k+1
    low_q = mod(floor(double(low_seq(:)) * 10^10), 4096);
    high_q = mod(floor(double(high_seq(:)) * 10^10), 4096);
    lut = dna_12bit_lut();   % 12×4096
    pair_values = zeros(12, pixel_count, 'uint8');
    for p = 1:pixel_count
        pair_values(:, p) = uint8(2 * lut(:, high_q(p) + 1) + lut(:, low_q(p) + 1));
    end
end

function lut = dna_12bit_lut()
%DNA_12_BIT_LUT 4096×12 位展开表, 第 k 行 = 第 (11-k) 位 (MSB 在前)
    idx = uint16(0:4095);
    lut = zeros(12, 4096, 'uint8');
    for bit = 11:-1:0
        lut(12 - bit, :) = uint8(bitand(bitshift(idx, -bit), 1));
    end
end
