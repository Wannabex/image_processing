function [watermark] = get_lsb_watermark_from_gray(gray_image)
img_dimensions = size(gray_image);
row_max = img_dimensions(1);
col_max = img_dimensions(2);

watermark = zeros(row_max, col_max);
for row = 1:row_max    
    for col = 1:col_max       
        watermark(row, col) = bitget(gray_image(row,col), 1);        
    end        
end
end

