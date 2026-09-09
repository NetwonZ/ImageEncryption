function out = salomon_global_bit_permutation_v2(dna_matrix, key)
%SALOMON_GLOBAL_BIT_PERMUTATION_V2 Forward coupled A/B/C circular permutation.
% Matches Encrypter.global_bit_permutation_v2 in Encryption.py.
    [P, H, W] = size(dna_matrix);
    if P ~= 12, error('DNA matrix must have 12 planes'); end
    out = zeros(P, H, W, 'uint8');
    sb = double(key.shift_b);  % H x 12
    sc = double(key.shift_c);  % W x 12
    sa = double(key.shift_a);  % H x W
    for row = 1:H
        for col = 1:W
            for plane = 1:P
                source_row = mod(row - 1 - sc(col, plane), H) + 1;
                source_col = mod(col - 1 - sb(source_row, plane), W) + 1;
                source_plane = mod(plane - 1 - sa(source_row, source_col), P) + 1;
                out(plane, row, col) = dna_matrix(source_plane, source_row, source_col);
            end
        end
    end
end
