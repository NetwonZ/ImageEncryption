function result = salomon_decrypt(cipher_input, km, blocks, varargin)
%SALOMON_DECRYPT Reverse salomon_encrypt using generated key material.
    parser = inputParser;
    addParameter(parser, 'parallel_size', []);
    parse(parser, varargin{:});
    parallel_size = parser.Results.parallel_size;
    if isempty(parallel_size)
        if isfield(km, 'global_parallel_size')
            parallel_size = km.global_parallel_size;
        else
            parallel_size = 1;
        end
    end

    if ischar(cipher_input) || isstring(cipher_input)
        image = imread(char(cipher_input));
    else
        image = cipher_input;
    end
    if ndims(image) == 2 || size(image, 3) == 1
        image = repmat(image(:, :, 1), 1, 1, 3);
    elseif size(image, 3) > 3
        image = image(:, :, 1:3);
    end
    image = uint8(image);

    cipher_bitplanes = image_to_bitplanes(image);
    cipher_dna = encode_to_dna(cipher_bitplanes, blocks, km.shuffled_rule_ids);
    locally_diffused = salomon_global_diffusion(cipher_dna, km, parallel_size, false);
    globally_permuted = salomon_block_diffusion(locally_diffused, blocks, km.block_key_matrices, km.block_operation_rule_ids, false);
    permuted = salomon_invert_global_bit_permutation_v2(globally_permuted, km);
    original_dna = salomon_invert_intra_block_permutation(permuted, blocks, km.intra_permutations);
    original_bits = decode_from_dna(original_dna, blocks, km.EncodeRule);
    result = struct('decrypted_image', bitplanes_to_image(original_bits));
end
