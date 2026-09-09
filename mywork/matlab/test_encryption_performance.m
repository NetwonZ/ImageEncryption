% Test encryption/decryption execution time by stage and percentage.
clear; clc;

image_path = '../../images/img3.png';
b_max = 64;
b_min = 12;
parallel_size = 1;

% ---------------- Encryption ----------------
stages = {};
times = [];

started = tic;
image = imread(image_path);
if ndims(image) == 2 || size(image, 3) == 1
    image = repmat(image(:, :, 1), 1, 1, 3);
elseif size(image, 3) > 3
    image = image(:, :, 1:3);
end
image = uint8(image);
[height, width, ~] = size(image);
stages{end + 1} = 'Image loading / RGB conversion';
times(end + 1) = toc(started);

started = tic;
blocks = salomon_adaptive_partition(image, b_max, b_min, []);
stages{end + 1} = 'Adaptive partition';
times(end + 1) = toc(started);

started = tic;
key_stream = salomon_keystream(height * width, image_path, []);
cml = struct('X', key_stream.X, 'mu', key_stream.mu, 'v', key_stream.v, ...
    'alpha', key_stream.alpha, 'beta', key_stream.beta, 'xi', 1, 'eta', 1);
stages{end + 1} = 'Key stream and CML initialization';
times(end + 1) = toc(started);

started = tic;
key_material = salomon_key_material(cml, blocks, height, width);
key_material.global_parallel_size = parallel_size;
stages{end + 1} = 'Key material generation';
times(end + 1) = toc(started);

started = tic;
bitplanes = image_to_bitplanes(image);
stages{end + 1} = 'RGB to bitplanes';
times(end + 1) = toc(started);

started = tic;
dna = encode_to_dna(bitplanes, blocks, key_material.EncodeRule);
stages{end + 1} = 'DNA encoding';
times(end + 1) = toc(started);

started = tic;
permuted = salomon_permute_within_blocks_V0(dna, blocks, key_material.intra_permutations);
stages{end + 1} = 'DNA intra-block permutation';
times(end + 1) = toc(started);

started = tic;
globally_permuted = salomon_global_bit_permutation_v2(permuted, key_material);
stages{end + 1} = 'Global A/B/C permutation';
times(end + 1) = toc(started);

started = tic;
locally_diffused = salomon_block_diffusion(globally_permuted, blocks, ...
    key_material.block_key_matrices, key_material.block_operation_rule_ids, true);
stages{end + 1} = 'Blockwise DNA diffusion';
times(end + 1) = toc(started);

started = tic;
globally_diffused = salomon_global_diffusion(locally_diffused, key_material, parallel_size, true);
stages{end + 1} = 'Global DNA diffusion';
times(end + 1) = toc(started);

started = tic;
encrypted_bitplanes = decode_from_dna(globally_diffused, blocks, key_material.shuffled_rule_ids);
stages{end + 1} = 'DNA decoding to bitplanes';
times(end + 1) = toc(started);

started = tic;
encrypted_image = bitplanes_to_image(encrypted_bitplanes);
stages{end + 1} = 'Bitplanes to RGB image';
times(end + 1) = toc(started);

print_profile('Encryption', stages, times);

% ---------------- Decryption ----------------
stages = {};
times = [];

started = tic;
cipher_bitplanes = image_to_bitplanes(encrypted_image);
stages{end + 1} = 'Cipher RGB to bitplanes';
times(end + 1) = toc(started);

started = tic;
cipher_dna = encode_to_dna(cipher_bitplanes, blocks, key_material.shuffled_rule_ids);
stages{end + 1} = 'DNA encoding';
times(end + 1) = toc(started);

started = tic;
locally_diffused = salomon_global_diffusion(cipher_dna, key_material, parallel_size, false);
stages{end + 1} = 'Inverse global DNA diffusion';
times(end + 1) = toc(started);

started = tic;
globally_permuted = salomon_block_diffusion(locally_diffused, blocks, ...
    key_material.block_key_matrices, key_material.block_operation_rule_ids, false);
stages{end + 1} = 'Inverse blockwise DNA diffusion';
times(end + 1) = toc(started);

started = tic;
permuted = salomon_invert_global_bit_permutation_v2(globally_permuted, key_material);
stages{end + 1} = 'Inverse global A/B/C permutation';
times(end + 1) = toc(started);

started = tic;
original_dna = salomon_invert_intra_block_permutation(permuted, blocks, key_material.intra_permutations);
stages{end + 1} = 'Inverse DNA intra-block permutation';
times(end + 1) = toc(started);

started = tic;
original_bitplanes = decode_from_dna(original_dna, blocks, key_material.EncodeRule);
stages{end + 1} = 'DNA decoding to bitplanes';
times(end + 1) = toc(started);

started = tic;
decrypted_image = bitplanes_to_image(original_bitplanes); %#ok<NASGU>
stages{end + 1} = 'Bitplanes to RGB image';
times(end + 1) = toc(started);

print_profile('Decryption', stages, times);

function print_profile(title_text, stages, times)
    total = sum(times);
    fprintf('\n%s performance profile\n', title_text);
    fprintf('%-42s %12s %10s\n', 'Stage', 'Time(s)', 'Percent');
    fprintf('%s\n', repmat('-', 1, 68));
    for index = 1:numel(stages)
        if total == 0
            percent = 0;
        else
            percent = times(index) / total * 100;
        end
        fprintf('%-42s %12.6f %9.2f%%\n', stages{index}, times(index), percent);
    end
    fprintf('%s\n', repmat('-', 1, 68));
    fprintf('%-42s %12.6f %9.2f%%\n', 'Total', total, 100);
end
