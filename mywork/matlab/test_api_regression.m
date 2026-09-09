% Regression test: public MATLAB interfaces with in-memory RGB data and custom key.
clear; clc;
H = 33; W = 35;
seed_image = uint8(mod(reshape(0:H * W * 3 - 1, H, W, 3), 256));
key = repmat('01', 1, 128);

first = salomon_encrypt(seed_image, 'b_max', 16, 'b_min', 8, 'parallel_size', 7);
second = salomon_encrypt(seed_image, 'b_max', 16, 'b_min', 8, 'parallel_size', 7);
plain_default = salomon_decrypt(first.encrypted_image, first.key_material, first.blocks);
custom = salomon_encrypt(seed_image, 'user_key', key, 'b_max', 16, 'b_min', 8, 'parallel_size', 7);
plain_custom = salomon_decrypt(custom.encrypted_image, custom.key_material, custom.blocks, 'parallel_size', 7);

fprintf('default deterministic: %d\n', isequal(first.encrypted_image, second.encrypted_image));
fprintf('default exact decrypt: %d\n', isequal(seed_image, plain_default.decrypted_image));
fprintf('custom exact decrypt: %d\n', isequal(seed_image, plain_custom.decrypted_image));
fprintf('custom key changes cipher: %d\n', ~isequal(first.encrypted_image, custom.encrypted_image));
fprintf('blocks: %d\n', numel(first.blocks));
assert(isequal(first.encrypted_image, second.encrypted_image));
assert(isequal(seed_image, plain_default.decrypted_image));
assert(isequal(seed_image, plain_custom.decrypted_image));
assert(~isequal(first.encrypted_image, custom.encrypted_image));
fprintf('API REGRESSION: PASS\n');
