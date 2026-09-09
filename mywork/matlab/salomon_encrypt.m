function result = salomon_encrypt(image_input, varargin)
%SALOMON_ENCRYPT Encrypt one RGB image with adaptive DNA/CML pipeline.
%   result = salomon_encrypt('image.png')
%   result = salomon_encrypt(rgb_uint8)
%   result = salomon_encrypt(..., 'user_key', key256, 'b_max', 64,
%       'b_min', 16, 'parallel_size', 64, 'save_path', 'cipher.png')
%
% Pipeline mirrors mywork/Encryption.py, global_bit_permutation_v2 mode:
% normalized RGB -> adaptive partition -> image/key CML -> key material ->
% bitplanes -> block DNA coding -> 3-axis block permutation -> A/B/C global
% permutation -> block DNA diffusion -> global ciphertext feedback ->
% DNA decoding -> encrypted RGB.
%
% image_input path hashes original file bytes. Array input hashes normalized
% RGB uint8 bytes in C-order with shape/dtype header. If user_key is omitted,
% shared fixed 256-bit default applies.

    parser = inputParser;
    addParameter(parser, 'user_key', []);
    addParameter(parser, 'b_max', 64);
    addParameter(parser, 'b_min', 12);
    addParameter(parser, 'parallel_size', 1);
    addParameter(parser, 'save_path', '');
    parse(parser, varargin{:});
    options = parser.Results;

    if ischar(image_input) || isstring(image_input)
        source_path = char(image_input);
        if ~isfile(source_path), error('plaintext image not found: %s', source_path); end
        image = imread(source_path);
        key_input = source_path;
    else
        image = image_input;
        key_input = [];
    end
    image = normalize_rgb_uint8(image);
    [height, width, ~] = size(image);
    if height == 0 || width == 0, error('plaintext image is empty'); end
    if options.b_max <= 0 || options.b_min <= 0 || options.b_min > options.b_max
        error('b_max and b_min must be positive and b_min must not exceed b_max');
    end

    if isempty(key_input)
        key_input = image;
    end
    blocks = salomon_adaptive_partition(image, options.b_max, options.b_min, []);
    key_stream = salomon_keystream(height * width, key_input, options.user_key);
    cml = struct('X', key_stream.X, 'mu', key_stream.mu, 'v', key_stream.v, ...
        'alpha', key_stream.alpha, 'beta', key_stream.beta, 'xi', 1, 'eta', 1);
    key_material = salomon_key_material(cml, blocks, height, width);
    key_material.global_parallel_size = options.parallel_size;

    bitplanes = image_to_bitplanes(image);
    dna = encode_to_dna(bitplanes, blocks, key_material.EncodeRule);
    block_permuted = salomon_permute_within_blocks_V0(dna, blocks, key_material.intra_permutations);
    globally_permuted = salomon_global_bit_permutation_v2(block_permuted, key_material);
    locally_diffused = salomon_block_diffusion(globally_permuted, blocks, ...
        key_material.block_key_matrices, key_material.block_operation_rule_ids, true);
    globally_diffused = salomon_global_diffusion(locally_diffused, key_material, options.parallel_size, true);
    encrypted_bits = decode_from_dna(globally_diffused, blocks, key_material.shuffled_rule_ids);
    encrypted_image = bitplanes_to_image(encrypted_bits);

    result = struct();
    result.encrypted_image = encrypted_image;
    result.key_material = key_material;
    result.blocks = blocks;
    result.meta = struct('version', 'encryption-v4-matlab', ...
        'shape', [height, width, 3], 'b_max', options.b_max, ...
        'b_min', options.b_min, 'parallel_size', options.parallel_size, ...
        'global_permutation_method', 'global_bit_permutation_v2');
    if ~isempty(options.save_path)
        imwrite(encrypted_image, char(options.save_path));
    end
end

function image = normalize_rgb_uint8(image)
    if ndims(image) == 2 || size(image, 3) == 1
        image = repmat(image(:, :, 1), 1, 1, 3);
    elseif size(image, 3) >= 3
        image = image(:, :, 1:3);
    else
        error('image must be 2-D grayscale or a 3-D array with channels');
    end
    if ~isa(image, 'uint8')
        values = double(image);
        if ~isempty(values) && all(isfinite(values(:))) && max(values(:)) <= 1.0
            values = values * 255.0;
        end
        values = min(max(values, 0), 255);
        image = uint8(values);
    end
end
