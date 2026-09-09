function out = dna_apply_op(left, right, operation)
%DNA_APPLY_OP elementwise DNA operation on two 12xHxW matrices (values 0..3).
    left = uint8(left); right = uint8(right);
    switch lower(operation)
        case 'xor'
            out = bitxor(left, right);
        case 'add'
            out = uint8(mod(int32(left) + int32(right), 4));
        case 'sub'
            out = uint8(mod(int32(left) - int32(right), 4));
        otherwise
            error('operation must be xor/add/sub');
    end
end
