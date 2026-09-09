function out = salomon_invert_intra_block_permutation(dna_matrix, blocks, intra_permutations)
%SALOMON_INVERT_INTRA_BLOCK_PERMUTATION Reverse block plane/row/column permutations.
    out = uint8(dna_matrix);
    for b = 1:numel(blocks)
        blk = blocks(b);
        r0 = blk.row + 1; c0 = blk.col + 1;
        h = blk.height; w = blk.width;
        pp = inverse_permutation(intra_permutations{b}.plane);
        rp = inverse_permutation(intra_permutations{b}.row);
        cp = inverse_permutation(intra_permutations{b}.col);
        sub = dna_matrix(:, r0:r0 + h - 1, c0:c0 + w - 1);
        out(:, r0:r0 + h - 1, c0:c0 + w - 1) = sub(pp, rp, cp);
    end
end

function p = inverse_permutation(perm)
    p = zeros(size(perm), 'like', perm);
    for k = 1:numel(perm)
        p(perm(k)) = k;
    end
end
