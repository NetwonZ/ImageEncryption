function out = dna_rule_table()
%DNA_RULE_TABLE 8 DNA coding rules (matches Python DNA_RULE_TABLE).
%   DNA_RULE_TABLE(rule, value+1) = base index (0..3 = A,G,C,T)
    out = uint8([
        0, 2, 1, 3;   % Rule 1: A G C T
        0, 1, 2, 3;   % Rule 2: A C G T
        2, 0, 3, 1;   % Rule 3: G A T C
        2, 3, 0, 1;   % Rule 4: G T A C
        1, 0, 3, 2;   % Rule 5: C A T G
        1, 3, 0, 2;   % Rule 6: C T A G
        3, 2, 1, 0;   % Rule 7: T G C A
        3, 1, 2, 0;   % Rule 8: T C G A
    ]);
end
