function image = bitplanes_to_image(bitplanes)
%BITPLANES_TO_IMAGE 24xHxW bitplanes -> RGB uint8.
    [~, H, W] = size(bitplanes);
    image = zeros(H, W, 3, 'uint8');
    plane = 0;
    for ch = 1:3
        acc = zeros(H, W, 'uint8');
        for bit = 7:-1:0
            plane = plane + 1;
            plane_bits = squeeze(uint8(bitplanes(plane, :, :)));
            acc = bitor(acc, bitand(plane_bits, 1) * uint8(2 ^ bit));
        end
        image(:, :, ch) = acc;
    end
end
