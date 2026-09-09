function dna = encode_to_dna(bitplanes, blocks, rule_ids)
%ENCODE_TO_DNA 24 bitplanes -> 12 DNA planes.
%   bitplanes: 24xHxW uint8 (0/1), per channel MSB first
%   blocks:    struct array {row,col,height,width}
%   rule_ids:  Bx1 uint8 (1..8)
%   dna:       12xHxW uint8 (0..3)
    T = double(dna_rule_table());
    [~, H, W] = size(bitplanes);
    dna = zeros(12, H, W, 'uint8');
    B = numel(blocks);
    for b = 1:B
        r = blocks(b).row + 1;
        c = blocks(b).col + 1;
        h = blocks(b).height;
        w = blocks(b).width;
        msb = bitplanes(1:2:24, r:r + h - 1, c:c + w - 1);
        lsb = bitplanes(2:2:24, r:r + h - 1, c:c + w - 1);
        values = double(msb) * 2 + double(lsb);
        rule = double(rule_ids(b));
        lookup = T(rule, values + 1);
        dna(:, r:r + h - 1, c:c + w - 1) = uint8(reshape(lookup, size(values)));
    end
end
