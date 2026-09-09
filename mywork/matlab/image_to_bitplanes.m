function bitplanes = image_to_bitplanes(image)
%IMAGE_TO_BITPLANES RGB uint8 -> 24xHxW bitplanes (per channel MSB first).
    image = uint8(image);
    [H, W, ~] = size(image);
    bitplanes = zeros(24, H, W, 'uint8');
    plane = 0;
    for ch = 1:3
        for bit = 7:-1:0
            plane = plane + 1;
            bitplanes(plane, :, :) = uint8(bitand(bitshift(int32(image(:, :, ch)), -bit), 1));
        end
    end
end
