function out = salomon_permute_within_blocks_V0(dna_matrix, blocks, intra_permutations)
%SALOMON_PERMUTE_WITHIN_BLOCKS_V0 Forward independent plane/row/column permutations.
% Mirrors Encrypter.permute_within_blocks_V0 in Encryption.py.
    out = uint8(dna_matrix);
    for b = 1:numel(blocks)
        blk = blocks(b);
        r0 = blk.row + 1; c0 = blk.col + 1;
        h = blk.height; w = blk.width;
        pp = intra_permutations{b}.plane;
        rp = intra_permutations{b}.row;
        cp = intra_permutations{b}.col;
        sub = dna_matrix(:, r0:r0 + h - 1, c0:c0 + w - 1);
        out(:, r0:r0 + h - 1, c0:c0 + w - 1) = sub(pp, rp, cp);
    end
end
