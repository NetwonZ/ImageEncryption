# MATLAB implementation of S-NCML adaptive DNA image encryption

## Interfaces

```matlab
result = salomon_encrypt('input.png');
plain = salomon_decrypt(result.encrypted_image, result.key_material, result.blocks);
assert(isequal(imread('input.png'), plain.decrypted_image));
```

`salomon_encrypt` accepts a file path or a grayscale/RGB array. Its defaults
match Python `EncryptionConfig`: `b_max=64`, `b_min=12`, and
`parallel_size=1`. It returns:

- `encrypted_image`: `H x W x 3` `uint8` ciphertext;
- `key_material`: CML-derived DNA rules, A/B/C shifts, diffusion keys, and global permutation;
- `blocks`: adaptive partition layout;
- `meta`: encryption parameters.

Use a caller-supplied 256-bit binary key when needed:

```matlab
key = repmat('01', 1, 128);
result = salomon_encrypt('input.png', 'user_key', key, ...
    'b_max', 64, 'b_min', 16, 'parallel_size', 64, ...
    'save_path', 'cipher.png');
plain = salomon_decrypt('cipher.png', result.key_material, result.blocks, ...
    'parallel_size', 64);
```

If `user_key` is omitted, the shared fixed 256-bit default from Python
`KeyStream.py` and `salomon_keystream.m` is used. Within MATLAB, same image,
key, and options produce same ciphertext deterministically.

## Implemented pipeline

1. RGB normalization and adaptive gray-gradient partition;
2. image hash plus user key to derive `mu`, `v`, `alpha`, `beta`, and lattice state;
3. Salomon coupled map lattice (CML) random-matrix generation;
4. 24 RGB bitplanes to 12 DNA planes using per-block coding rules;
5. per-block plane/row/column permutations;
6. coupled global A/B/C circular permutation;
7. per-block DNA XOR/add/sub diffusion and global ciphertext-feedback XOR diffusion;
8. inverse pipeline in `salomon_decrypt`.

Python and MATLAB retain same formulas and processing order. They need not yield
byte-identical ciphertext across runtimes because chaotic trajectories amplify
small platform-level floating-point differences in transcendental functions.

## Verification

Run from this directory:

```matlab
run('test_roundtrip.m')
run('test_api_regression.m')
```

Both tests require MATLAB Image Processing Toolbox functions `imread`/`imwrite`.
