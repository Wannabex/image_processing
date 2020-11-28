% Image processing experiment 2 - digital watermarking
% psnr_nbits_test.m - implementing n least significat bits watermark and
% measuring psnr
% 27.11.2020
% Krystian Lalik
clc; clear all; close all;

grayimg = imread("./lena512g.bmp");
nbits = 5;
max_watermark_bits = numel(grayimg) * nbits;

watermarkBits = rand(1, max_watermark_bits);
for i = 1:max_watermark_bits
    if watermarkBits(i) > 0.5
        watermarkBits(i) = logical(1);
    else
        watermarkBits(i) = logical(0);
    end
end

grayimg_watermarked = nbits_algorithm_gray(grayimg, watermarkBits, nbits);

subplot(1,2,1), imshow(grayimg);
subplot(1,2,2), imshow(grayimg_watermarked);

disp(psnr(grayimg_watermarked, grayimg))

