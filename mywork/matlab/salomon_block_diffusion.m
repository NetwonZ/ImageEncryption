function out = salomon_block_diffusion(dna_matrix, blocks, key_matrices, operation_rule_ids, is_encrypt)
%SALOMON_BLOCK_DIFFUSION Per-block DNA XOR/add/sub diffusion and inverse.
% Mirrors Encrypter.apply_block_diffusion and DeEncrypter.invert_block_diffusion.
    if nargin < 5 || isempty(is_encrypt), is_encrypt = true; end
    out = uint8(dna_matrix);
    if numel(key_matrices) ~= numel(blocks) || numel(operation_rule_ids) ~= numel(blocks)
        error('block key material does not match blocks');
    end
    for b = 1:numel(blocks)
        blk = blocks(b);
        r0 = blk.row + 1; c0 = blk.col + 1;
        h = blk.height; w = blk.width;
        switch operation_rule_ids(b)
            case 1, operation = 'xor';
            case 2, operation = 'add';
            case 3, operation = 'sub';
            otherwise, error('invalid block operation rule id');
        end
        source = dna_matrix(:, r0:r0 + h - 1, c0:c0 + w - 1);
        key_matrix = key_matrices{b};
        if is_encrypt
            out(:, r0:r0 + h - 1, c0:c0 + w - 1) = dna_apply_op(source, key_matrix, operation);
        else
            out(:, r0:r0 + h - 1, c0:c0 + w - 1) = dna_inv_op(source, key_matrix, operation);
        end
    end
end
