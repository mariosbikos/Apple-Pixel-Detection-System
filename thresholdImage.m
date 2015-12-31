function [ ThresholdedImg ] = thresholdImage( img,thresholdValue )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

ThresholdedImg=zeros(size(img,1),size(img,2));
for x=1:size(img,1)
    for y=1:size(img,2)
        ThresholdedImg(x,y)=(round(255*img(x,y)) >= thresholdValue);
    end
end


end

