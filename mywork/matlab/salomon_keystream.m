function key = salomon_keystream(L, img, user_key)
%SALOMON_KEYSTREAM Key-stream generation: derive chaos parameters and the
% initial state from image raw bytes + 256-bit user key.
% Mirrors mywork/KeyStream.py keystream_generation() (float path).
%
% Input:
%   L        scalar, number of lattice points (= H*W)
%   img      image path (char) or uint8 array (RGB/gray)
%   user_key 256-bit binary string; default = fixed DEFAULT_USER_KEY_BITS
% Output:
%   key      struct: mu, v, alpha, beta, X (Lx1 double)
%
% Steps (identical to Python):
%   1. plaintext_bytes = raw file bytes, or array header + raw bytes
%   2. sha256(plaintext_bytes) XOR user_key -> 256 bits H
%   3. H[250:256] -> g (6 bits); first 250 bits in 5 blocks of 50 -> V_j
%      U_j = |sin(4*pi*(V_j+g mod 1))| ; mu=10U0, v=10U1, alpha=10U2, beta=10U3, x=U4
%   4. iterate 1-D map L+warmup times, discard first warmup=50:
%      x = |sin((5+3*mu)*(1 - v*x*sin(15*pi*x*(1-x))))|
%
% Note: MATLAB sin may differ from Python math.sin by 1 ULP; chaos amplifies
% that, so ciphertexts are NOT bit-identical across languages. The flow and
% formulas are identical.

    if nargin < 3 || isempty(user_key)
        user_key = default_user_key_bits();
    end
    if ischar(user_key) || isstring(user_key)
        user_key = char(user_key);
    else
        error('user_key must be a 256-bit binary string');
    end
    if numel(user_key) ~= 256 || any(user_key ~= '0' & user_key ~= '1')
        error('user_key must contain exactly 256 bits of 0/1');
    end

    if ischar(img) || isstring(img)
        img = char(img);
        if ~isfile(img)
            error('plaintext image not found: %s', img);
        end
        fid = fopen(img, 'rb');
        plaintext_bytes = uint8(fread(fid, Inf, '*uint8'))';
        fclose(fid);
    else
        arr = uint8(img);
        sz = size(arr);
        dim_text = sprintf('%d, ', sz);
        dim_text = dim_text(1:end - 2);
        if numel(sz) == 1
            header = sprintf('shape=(%d,);dtype=|u1;', sz(1));
        else
            header = sprintf('shape=(%s);dtype=|u1;', dim_text);
        end
        % Python uses image_array.tobytes(order='C'). MATLAB storage is
        % column-major, so reverse dimensions before linearizing.
        c_order = permute(arr, ndims(arr):-1:1);
        plaintext_bytes = uint8([header, c_order(:).']);
    end
    if isempty(plaintext_bytes)
        error('plaintext image is empty');
    end

    % sha256 -> 256 bits, MSB first
    digest = sha256_bytes(plaintext_bytes);
    image_hash_bits = de2bi_uint8(digest, 8);   % column vector
    user_key_bits = (user_key == '1');
    user_key_bits = user_key_bits(:);           % column, same shape
    H = xor(image_hash_bits, user_key_bits);

    % parameter extraction
    g_int = bits_uint64(H(251:256));
    g = double(g_int) / 2^6;
    U = zeros(1, 5);
    for j = 1:5
        idx0 = (j - 1) * 50 + 1;
        block = H(idx0:idx0 + 49);
        V_j = double(bits_uint64(block)) / 2^50;
        mix_val = mod(V_j + g, 1.0);
        U(j) = abs(sin(4 * pi * mix_val));
    end
    key.mu = 10 * U(1);
    key.v = 10 * U(2);
    key.alpha = 10 * U(3);
    key.beta = 10 * U(4);

    % 1-D map iteration
    x = U(5);
    factor = 5.0 + 3.0 * key.mu;
    warmup = 50;
    total = L + warmup;
    X = zeros(L, 1);
    for step = 1:total
        x = abs(sin(factor * (1.0 - (key.v * x * sin(15.0 * pi * x * (1.0 - x))))));
        if step > warmup
            X(step - warmup) = x;
        end
    end
    key.X = X;
end

function bits = de2bi_uint8(bytes, nbits)
% expand each byte to nbits bits, MSB first (matches np.unpackbits bitorder='big')
    n = numel(bytes);
    bits = false(n * nbits, 1);
    for k = 1:n
        b = double(bytes(k));
        for j = 1:nbits
            bits((k - 1) * nbits + j) = bitand(bitshift(b, -(nbits - j)), 1) == 1;
        end
    end
end

function v = bits_uint64(bits)
% interpret logical bit vector as uint64 like Python sum(bit<<idx) with
% enumerate over MSB-first array: bit[0] gets shift 0 (LSB), bit[end] gets
% the highest shift. So the vector is effectively reversed.
    v = uint64(0);
    n = numel(bits);
    for k = 1:n
        if bits(k)
            v = v + bitshift(uint64(1), uint64(k - 1));
        end
    end
end

function s = default_user_key_bits()
% fixed 256-bit key, identical to mywork/KeyStream.py DEFAULT_USER_KEY_BITS
    s = [ ...
        '1111001111110100100100100100100111011100001010001111111110010000', ...
        '1010010110101110110001111001011110000011000001101101000000111011', ...
        '1111001110001011001011111111110010000000101001001101111101011010', ...
        '0101000111001001101111000111000000011110011111101010010000011001'];
end
