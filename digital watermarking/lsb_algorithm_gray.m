function [image_with_watermark] = lsb_algorithm_gray(gray_image, watermark)

img_dimensions = size(gray_image);
watermark_size = length(watermark);

row_max = img_dimensions(1);
col_max = img_dimensions(2);
row = 1;
col = 1;
current_bit = 1;
while current_bit < (row_max * col_max) && current_bit < watermark_size            
    gray_image(row, col) = bitset(gray_image(row, col), 1, watermark(current_bit));
    current_bit = current_bit + 1;    
    if mod(current_bit, col_max) == 0
        col = 1;
        row = row + 1;
    else
        col = col + 1;
    end               
end

image_with_watermark = gray_image;
end

