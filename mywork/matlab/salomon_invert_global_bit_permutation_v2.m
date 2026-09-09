function out = salomon_invert_global_bit_permutation_v2(dna_matrix, key)
%SALOMON_INVERT_GLOBAL_BIT_PERMUTATION_V2 Reverse coupled A/B/C permutation.
% Matches DeEncrypter.invert_global_bit_permutation_v2 in Encryption.py.
    [P, H, W] = size(dna_matrix);
    if P ~= 12, error('DNA matrix must have 12 planes'); end
    out = zeros(P, H, W, 'uint8');
    sb = double(key.shift_b);
    sc = double(key.shift_c);
    sa = double(key.shift_a);
    for row = 1:H
        for col = 1:W
            for plane = 1:P
                target_plane = mod(plane - 1 + sa(row, col), P) + 1;
                target_col = mod(col - 1 + sb(row, target_plane), W) + 1;
                target_row = mod(row - 1 + sc(target_col, target_plane), H) + 1;
                % For each original source coordinate, gather the value from
                % its forward target coordinate in the permuted tensor.
                out(plane, row, col) = dna_matrix(target_plane, target_row, target_col);
            end
        end
    end
end
