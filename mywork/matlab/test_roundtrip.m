% E2E test: encrypt img3.png, decrypt, verify roundtrip exactness.
clear; clc;
img = '../../images/img3.png';
res = salomon_encrypt(img, 'b_max', 64, 'b_min', 16, 'parallel_size', 64);
fprintf('blocks: %d (Python b_min=16 reference: 217)\n', numel(res.blocks));
fprintf('encrypted shape: %d x %d x %d\n', size(res.encrypted_image, 1), size(res.encrypted_image, 2), size(res.encrypted_image, 3));

dec = salomon_decrypt(res.encrypted_image, res.key_material, res.blocks, 'parallel_size', res.meta.parallel_size);
orig = imread(img);
if size(orig, 3) == 1, orig = repmat(orig, 1, 1, 3); end
orig = uint8(orig(:, :, 1:3));

diff = double(dec.decrypted_image) - double(orig);
fprintf('roundtrip max abs diff: %d\n', max(abs(diff(:))));
if max(abs(diff(:))) == 0
    fprintf('ROUNDTRIP EXACT: PASS\n');
else
    fprintf('ROUNDTRIP EXACT: FAIL\n');
    [r, c, ch] = ind2sub(size(diff), find(abs(diff) > 0, 1));
    fprintf('first diff at (%d,%d,%d): orig %d dec %d\n', r, c, ch, orig(r, c, ch), dec.decrypted_image(r, c, ch));
end
