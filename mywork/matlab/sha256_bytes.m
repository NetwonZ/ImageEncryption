function h = sha256_bytes(data)
%SHA256_BYTES SHA-256 digest (32 bytes) of a uint8 array. Pure MATLAB.
% Matches Python hashlib.sha256 (FIPS 180-4).
% NOTE: MATLAB integer arithmetic saturates; SHA needs mod-2^32 wraparound,
% so all additions are done in double with explicit mod 2^32.
    if ischar(data) || isstring(data)
        data = uint8(char(data));
    else
        data = uint8(data(:));
    end
    bitlen = uint64(numel(data)) * 8;
    n = numel(data);
    k = mod(56 - mod(n + 1, 64), 64);
    padded = [data; uint8(128); zeros(k, 1, 'uint8')];
    padded = [padded; typecast(swapbytes(bitlen), 'uint8')'];

    K = uint32([...
        0x428a2f98 0x71374491 0xb5c0fbcf 0xe9b5dba5 0x3956c25b 0x59f111f1 0x923f82a4 0xab1c5ed5 ...
        0xd807aa98 0x12835b01 0x243185be 0x550c7dc3 0x72be5d74 0x80deb1fe 0x9bdc06a7 0xc19bf174 ...
        0xe49b69c1 0xefbe4786 0x0fc19dc6 0x240ca1cc 0x2de92c6f 0x4a7484aa 0x5cb0a9dc 0x76f988da ...
        0x983e5152 0xa831c66d 0xb00327c8 0xbf597fc7 0xc6e00bf3 0xd5a79147 0x06ca6351 0x14292967 ...
        0x27b70a85 0x2e1b2138 0x4d2c6dfc 0x53380d13 0x650a7354 0x766a0abb 0x81c2c92e 0x92722c85 ...
        0xa2bfe8a1 0xa81a664b 0xc24b8b70 0xc76c51a3 0xd192e819 0xd6990624 0xf40e3585 0x106aa070 ...
        0x19a4c116 0x1e376c08 0x2748774c 0x34b0bcb5 0x391c0cb3 0x4ed8aa4a 0x5b9cca4f 0x682e6ff3 ...
        0x748f82ee 0x78a5636f 0x84c87814 0x8cc70208 0x90befffa 0xa4506ceb 0xbef9a3f7 0xc67178f2]);

    H0 = uint32([0x6a09e667 0xbb67ae85 0x3c6ef372 0xa54ff53a ...
                 0x510e527f 0x9b05688c 0x1f83d9ab 0x5be0cd19]);

    nblocks = numel(padded) / 64;
    for blk = 0:nblocks - 1
        chunk = padded(blk * 64 + 1:blk * 64 + 64);
        w = zeros(1, 64, 'uint32');
        for t = 0:15
            b = chunk(t * 4 + 1:t * 4 + 4);
            w(t + 1) = uint32(b(1)) * 16777216 + uint32(b(2)) * 65536 + uint32(b(3)) * 256 + uint32(b(4));
        end
        for t = 16:63
            s0 = bitxor(bitxor(rotl32(w(t - 15 + 1), 25), rotl32(w(t - 15 + 1), 14)), bitshift(w(t - 15 + 1), -3));
            s1 = bitxor(bitxor(rotl32(w(t - 2 + 1), 15), rotl32(w(t - 2 + 1), 13)), bitshift(w(t - 2 + 1), -10));
            w(t + 1) = wrap32(double(w(t - 16 + 1)) + double(s0) + double(w(t - 7 + 1)) + double(s1));
        end
        a = H0(1); b = H0(2); c = H0(3); d = H0(4); e = H0(5); f = H0(6); g = H0(7); h = H0(8);
        for t = 0:63
            S1 = bitxor(bitxor(rotl32(e, 26), rotl32(e, 21)), rotl32(e, 7));
            ch = bitxor(bitand(e, f), bitand(bitcmp(e), g));
            temp1 = wrap32(double(h) + double(S1) + double(ch) + double(K(t + 1)) + double(w(t + 1)));
            S0 = bitxor(bitxor(rotl32(a, 30), rotl32(a, 19)), rotl32(a, 10));
            maj = bitxor(bitxor(bitand(a, b), bitand(a, c)), bitand(b, c));
            temp2 = wrap32(double(S0) + double(maj));
            h = g; g = f; f = e;
            e = wrap32(double(d) + double(temp1));
            d = c; c = b; b = a;
            a = wrap32(double(temp1) + double(temp2));
        end
        H0(1) = wrap32(double(H0(1)) + double(a));
        H0(2) = wrap32(double(H0(2)) + double(b));
        H0(3) = wrap32(double(H0(3)) + double(c));
        H0(4) = wrap32(double(H0(4)) + double(d));
        H0(5) = wrap32(double(H0(5)) + double(e));
        H0(6) = wrap32(double(H0(6)) + double(f));
        H0(7) = wrap32(double(H0(7)) + double(g));
        H0(8) = wrap32(double(H0(8)) + double(h));
    end

    out = zeros(1, 32, 'uint8');
    for i = 1:8
        v = H0(i);
        out(i * 4 - 3) = uint8(bitand(bitshift(v, -24), 255));
        out(i * 4 - 2) = uint8(bitand(bitshift(v, -16), 255));
        out(i * 4 - 1) = uint8(bitand(bitshift(v, -8), 255));
        out(i * 4) = uint8(bitand(v, 255));
    end
    h = out;
end

function r = rotl32(x, n)
    r = bitor(bitshift(x, n), bitshift(x, n - 32));
end

function u = wrap32(v)
    u = uint32(mod(v, 4294967296));
end
