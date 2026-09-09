function bitplanes = decode_from_dna(dna_matrix, blocks, rule_ids)
%DECODE_FROM_DNA 12 DNA planes -> 24 bitplanes.
    invT = double(dna_rule_table_inv());
    [~, H, W] = size(dna_matrix);
    bitplanes = zeros(24, H, W, 'uint8');
    B = numel(blocks);
    for b = 1:B
        r = blocks(b).row + 1;
        c = blocks(b).col + 1;
        h = blocks(b).height;
        w = blocks(b).width;
        encoded = dna_matrix(:, r:r + h - 1, c:c + w - 1);
        lookup = invT(double(rule_ids(b)), double(encoded) + 1);
        vals = uint8(reshape(lookup, size(encoded)));
        bitplanes(1:2:24, r:r + h - 1, c:c + w - 1) = bitshift(vals, -1);
        bitplanes(2:2:24, r:r + h - 1, c:c + w - 1) = bitand(vals, 1);
    end
end
