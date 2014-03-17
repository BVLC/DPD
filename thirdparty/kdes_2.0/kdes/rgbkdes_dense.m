function feaSet = rgbkdes_dense(im, rgbkdes_params)
% dense color kernel descriptors over uniform grids sampled from images

% inputs
% im             -RGB image
% rgbkdes_params -parameters of color kernel descriptors
%
% outputs
% feaSet         -color kernel descriptors and their locations
% written by Liefeng Bo on 2010
% modified by Liefeng Bo on March 26, 2012
%==========================================================================


if ~isfield(rgbkdes_params,'grid'); rgbkdes_params.grid = 8; end
if ~isfield(rgbkdes_params,'patchsize'); rgbkdes_params.patchsize = 16; end
if ~isfield(rgbkdes_params,'kdesdim'); rgbkdes_params.kdesdim = 50; end
if ~isfield(rgbkdes_params,'mask'); rgbkdes_params.mask = ones(size(im,1),size(im,2)); end

% default setting
if rgbkdes_params.kdesdim ~= 50
   rgbkdes_params.kdes.eigvectors = rgbkdes_params.kdes.eigvectors(1:rgbkdes_params.kdesdim,:);
end

im = im2double(im); % normalize pixel values to [0 1]
if size(im,3) == 1
    im = color(im);
end

% densely and uniformly sample interest points
mpatchsize = max(rgbkdes_params.patchsize); % maximum patch size
[im_h, im_w, rgb] = size(im);
rem_x = mod(im_w-mpatchsize, rgbkdes_params.grid);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatchsize, rgbkdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:rgbkdes_params.grid:im_w-mpatchsize+1, offset_y:rgbkdes_params.grid:im_h-mpatchsize+1);
num_patches = numel(grid_x);

im_k = kernel_trans(im, rgbkdes_params);
% color basis vector for all pixels
for j = 1:length(rgbkdes_params.patchsize)
    xx = repmat((0:rgbkdes_params.patchsize(j)-1)/rgbkdes_params.patchsize(j), [rgbkdes_params.patchsize(j) 1]); % horizontal spatial position
    yy = repmat(((0:rgbkdes_params.patchsize(j)-1)')/rgbkdes_params.patchsize(j),[1 rgbkdes_params.patchsize(j)]); % vertical spatial position
    
    % spatial basis vectors
    skv = eval_kernel(rgbkdes_params.kdes.spoints', [yy(:) xx(:)],rgbkdes_params.kdes.ktype,rgbkdes_params.kdes.kparam(end-1:end));
    mwkvs = zeros(size(rgbkdes_params.kdes.eigvectors,2), num_patches);

    keep = [];
    it = 0;
    % for all patches
    for i = 1:num_patches
        
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + rgbkdes_params.patchsize(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + rgbkdes_params.patchsize(j) - 1;  
        
        submask = rgbkdes_params.mask(y_lo:y_hi,x_lo:x_hi);
        if sum(submask(:)) > rgbkdes_params.patchsize(j)*sqrt(rgbkdes_params.patchsize(j))
           keep = [keep i];
           it = it + 1;
           % compute kernel descriptor
           mwkv = (skv*reshape( im_k(y_lo:y_hi,x_lo:x_hi,:),rgbkdes_params.patchsize(j)^2, rgbkdes_params.kdes.rgbsize))'/size(skv,2);
           mwkvs(:,i) = mwkv(:);
        end
    end
    rgbkdes_arr = rgbkdes_params.kdes.eigvectors*mwkvs;
    rgbkdes_arr(:,it+1:end) = [];
    feaSet.feaArr{j} = rgbkdes_arr;
end

% output feature information
feaSet.x = grid_x(keep) + mpatchsize/2 - 0.5;
feaSet.y = grid_y(keep) + mpatchsize/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function cim = color(im)

cim(:,:,1) = im;
cim(:,:,2) = im;
cim(:,:,3) = im;

function I_k = kernel_trans(I, rgbkdes_params)

[asize,bsize,rgbsize] = size(I);
I_rgb = reshape(I,asize*bsize,rgbsize);
I_k = eval_kernel(I_rgb, rgbkdes_params.kdes.rgbpoints', rgbkdes_params.kdes.ktype, rgbkdes_params.kdes.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,rgbkdes_params.kdes.rgbsize);


