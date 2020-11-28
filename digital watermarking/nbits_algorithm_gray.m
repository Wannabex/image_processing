function [image_with_watermark] = nbits_algorithm_gray(gray_image, watermark, nbits)

img_dimensions = size(gray_image);
watermark_size = length(watermark);

row_max = img_dimensions(1);
col_max = img_dimensions(2);
row = 1;
col = 1;
current_pixel = 1;
while current_pixel < (row_max * col_max) && current_pixel < watermark_size            
    for i = 1:nbits
        gray_image(row, col) = bitset(gray_image(row, col), i, watermark((current_pixel-1) *nbits + i));
    end    
    current_pixel = current_pixel + 1;    
    if mod(current_pixel, col_max) == 0
        col = 1;
        row = row + 1;
    else
        col = col + 1;
    end               
end

image_with_watermark = gray_image;
end

