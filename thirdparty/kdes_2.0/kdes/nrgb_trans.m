function [im, im_v] = nrgb_trans(im)
% normalize rgb values by mean and standard deviation
% written by Liefeng Bo on 2012

% mean filter
psize = 5;
mfilter = ones(psize)/psize^2;

% subtract mean
im_m = sum(im,3)/size(im,3);
im_m = filter2(mfilter, im_m, 'same');
for i = 1:size(im,3)
  im(:,:,i) = im(:,:,i) - im_m;
end 

% compute standard deviation
im_v = sum(im.^2, 3);
im_v = sqrt(filter2(mfilter, im_v, 'same') + 0.001);

% normalize magnitude by standard deviation
for i = 1:size(im,3)
  im(:,:,i) = im(:,:,i)./im_v/4;
end
 
