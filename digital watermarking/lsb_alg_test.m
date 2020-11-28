% Image processing experiment 2 - digital watermarking
% lsb_alg_test.m - testing LSB algorithm for grayscale and colour image
% 26.11.2020
% Krystian Lalik
clc; clear all; close all;

grayimg = imread("./lena512g.bmp");
colorimg = imread("./lena512.bmp");
watermark = 'Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian, Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian KrystianKrystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian Krystian';
watermarkBits = logical(reshape(transpose(dec2bin(watermark,8)), 1, []));

grayimg_watermarked = lsb_algorithm_gray(grayimg, watermarkBits);
colorimg_watermarked = lsb_algorithm_color(colorimg, watermarkBits);

subplot(2,3,1), imshow(grayimg);
subplot(2,3,2), imshow(grayimg_watermarked);
subplot(2,3,3), imshow(imsubtract(grayimg_watermarked, grayimg));
subplot(2,3,4), imshow(colorimg);
subplot(2,3,5), imshow(colorimg_watermarked);
subplot(2,3,6), imshow(imsubtract(colorimg_watermarked, colorimg));

