function out = dna_inv_op(cipher, key, operation)
%DNA_INV_OP inverse DNA operation.
    cipher = uint8(cipher); key = uint8(key);
    switch lower(operation)
        case 'xor'
            out = bitxor(cipher, key);
        case 'add'
            out = uint8(mod(int32(cipher) - int32(key), 4));
        case 'sub'
            out = uint8(mod(int32(cipher) + int32(key), 4));
        otherwise
            error('operation must be xor/add/sub');
    end
end
