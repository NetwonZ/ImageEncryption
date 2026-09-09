function out = salomon_global_diffusion(dna_matrix, key, parallel_size, is_encrypt)
%SALOMON_GLOBAL_DIFFUSION Global grouped ciphertext-feedback XOR diffusion.
% is_encrypt=false reconstructs plaintext from ciphertext using same recurrence.
    if nargin < 3 || isempty(parallel_size), parallel_size = 1; end
    if nargin < 4 || isempty(is_encrypt), is_encrypt = true; end
    if ~isscalar(parallel_size) || parallel_size < 1 || parallel_size ~= floor(parallel_size)
        error('parallel_size must be a positive integer');
    end
    [P, H, W] = size(dna_matrix);
    num_pixels = H * W;
    perm = double(key.global_permutation(:)).';
    if numel(perm) ~= num_pixels || any(sort(perm) ~= 1:num_pixels)
        error('global permutation is invalid');
    end

    flat = zeros(P, num_pixels, 'uint8');
    key_flat = zeros(P, num_pixels, 'uint8');
    for pos = 1:num_pixels
        row = floor((pos - 1) / W) + 1;
        col = mod(pos - 1, W) + 1;
        flat(:, pos) = dna_matrix(:, row, col);
        key_flat(:, pos) = key.global_key_matrix(:, row, col);
    end
    input_sequence = flat(:, perm);
    key_sequence = key_flat(:, perm);
    output_sequence = zeros(P, num_pixels, 'uint8');
    previous = uint8(key.global_initial_vector(:));
    for start = 1:parallel_size:num_pixels
        stop = min(start + parallel_size - 1, num_pixels);
        output_sequence(:, start:stop) = bitxor( ...
            bitxor(input_sequence(:, start:stop), previous), ...
            key_sequence(:, start:stop));
        if is_encrypt
            % C_i becomes predecessor for following group.
            previous = output_sequence(:, stop);
        else
            % During inversion, predecessor is prior ciphertext, not the
            % recovered plaintext.
            previous = input_sequence(:, stop);
        end
    end

    output_flat = zeros(P, num_pixels, 'uint8');
    output_flat(:, perm) = output_sequence;
    out = zeros(P, H, W, 'uint8');
    for pos = 1:num_pixels
        row = floor((pos - 1) / W) + 1;
        col = mod(pos - 1, W) + 1;
        out(:, row, col) = output_flat(:, pos);
    end

    % Encryption and decryption differ only in whether input_sequence is
    % plaintext or ciphertext; recurrence above is the same algebraically.
    if ~is_encrypt
        out = uint8(out);
    end
end
