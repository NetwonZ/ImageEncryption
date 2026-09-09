function out = dna_rule_table_inv()
%DNA_RULE_TABLE_INV inverse rule table: inv(rule, base+1) = 2-bit value.
% Matches np.argsort(DNA_RULE_TABLE, axis=1).
    T = double(dna_rule_table());
    invT = zeros(8, 4, 'uint8');
    for r = 1:8
        for v = 0:3
            base = T(r, v + 1);
            invT(r, base + 1) = v;
        end
    end
    out = invT;
end
