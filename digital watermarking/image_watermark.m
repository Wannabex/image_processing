% Image processing experiment 2 - digital watermarking
% image_watermark.m - trying to embed image as watermark into another image
% 26.11.2020
% Krystian Lalik
clc; clear all; close all;

grayimg = imread("./lena512g.bmp");
watermarkimg = im2bw(imread("zwierze.jpg"), 0.5);
watermarkBits = watermarkimg';
watermarkBits = watermarkBits(:)';

grayimg_watermarked = lsb_algorithm_gray(grayimg, watermarkBits);

% Task 3 - manipulation of an image with watermark
grayimg_watermarked_cropped = imcrop(grayimg_watermarked, [1, 1,512,348]);
grayimg_watermarked_cropped = imresize(grayimg_watermarked_cropped, 0.9);
grayimg_watermarked = imrotate(grayimg_watermarked, 90);
% Task 3 end

watermark_cropped = get_lsb_watermark_from_gray(grayimg_watermarked_cropped);
watermark = get_lsb_watermark_from_gray(grayimg_watermarked);

subplot(3,2,1), imshow(grayimg);
subplot(3,2,2), imshow(watermarkimg);
subplot(3,2,3), imshow(grayimg_watermarked);
subplot(3,2,4), imshow(watermark);
% Task 3 manipulated images
subplot(3,2,5), imshow(grayimg_watermarked_cropped);
subplot(3,2,6), imshow(watermark_cropped);

disp(psnr(grayimg_watermarked, grayimg))

