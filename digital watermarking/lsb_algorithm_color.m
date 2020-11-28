function [image_with_watermark] = lsb_algorithm_color(color_image, watermark)
img_dimensions = size(color_image);
watermark_size = length(watermark);

row_max = img_dimensions(1);
col_max = img_dimensions(2);
channel_max = img_dimensions(3);
row = 1;
col = 1;
channel = 1;
current_bit = 1;
while current_bit < (row_max * col_max) && current_bit < watermark_size          
    color_image(row, col, channel) = bitset(color_image(row, col, channel), 1, str2double(watermark(current_bit)));
    current_bit = current_bit + 1;            
    if mod(current_bit, channel_max) == 0
        channel = 1;                
        if mod(current_bit, (col_max * channel_max)) == 0
        col = 1;
        row = row + 1;
        else
            col = col + 1;
        end                
    else
        channel = channel + 1;
    end                               
end

image_with_watermark = color_image;
end
