function feaSet = nrgbkdes_dense(im, nrgbkdes_params)
% dense normalized color kernel descriptors over uniform grids sampled from images
% robust to lighting condition changes

% inputs
% im			-RGB image
% nrgbkdes_params	-parameters of normalized color kernel descriptors
%
% outputs
% feaSet		-normalized color kernel descriptors and their locations
% written by Liefeng Bo on 2012

%==========================================================================

if ~isfield(nrgbkdes_params,'grid'); nrgbkdes_params.grid = 8; end
if ~isfield(nrgbkdes_params,'patchsize'); nrgbkdes_params.patchsize = 16; end
if ~isfield(nrgbkdes_params,'kdesdim'); nrgbkdes_params.kdesdim = 50; end
if ~isfield(nrgbkdes_params,'contrast'); nrgbkdes_params.contrast = 0.8; end
if ~isfield(nrgbkdes_params,'mask'); nrgbkdes_params.mask = ones(size(im,1),size(im,2)); end

% default setting
if nrgbkdes_params.kdesdim ~= 50
   nrgbkdes_params.kdes.eigvectors = nrgbkdes_params.kdes.eigvectors(1:nrgbkdes_params.kdesdim,:);
end

im = im2double(im); % normalize pixel values to [0 1]
if size(im,3) == 1
    im = color(im);
end

[im, im_v] = nrgb_trans(im);

% densely and uniformly sample interest points
mpatchsize = max(nrgbkdes_params.patchsize); % maximum patch size
[im_h, im_w, rgb] = size(im);
rem_x = mod(im_w-mpatchsize, nrgbkdes_params.grid);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatchsize, nrgbkdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:nrgbkdes_params.grid:im_w-mpatchsize+1, offset_y:nrgbkdes_params.grid:im_h-mpatchsize+1);
num_patches = numel(grid_x);

im_k = kernel_trans(im, nrgbkdes_params);
% color basis vector for all pixels
for j = 1:length(nrgbkdes_params.patchsize)
    xx = repmat((0:nrgbkdes_params.patchsize(j)-1)/nrgbkdes_params.patchsize(j), [nrgbkdes_params.patchsize(j) 1]); % horizontal spatial position
    yy = repmat(((0:nrgbkdes_params.patchsize(j)-1)')/nrgbkdes_params.patchsize(j),[1 nrgbkdes_params.patchsize(j)]); % vertical spatial position
    
    % spatial basis vectors
    skv = eval_kernel(nrgbkdes_params.kdes.spoints', [yy(:) xx(:)],nrgbkdes_params.kdes.ktype, nrgbkdes_params.kdes.kparam(end-1:end));
    mwkvs = zeros(size(nrgbkdes_params.kdes.eigvectors,2), num_patches);

    keep = [];
    it = 0;
    % for all patches
    for i = 1:num_patches
        
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + nrgbkdes_params.patchsize(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + nrgbkdes_params.patchsize(j) - 1;  
        
        submask = nrgbkdes_params.mask(y_lo:y_hi,x_lo:x_hi);
        if sum(submask(:)) > nrgbkdes_params.patchsize(j)*sqrt(nrgbkdes_params.patchsize(j))
           keep = [keep i];
           it = it + 1;

           % standard deviation
           weight = im_v(y_lo:y_hi,x_lo:x_hi);
           weight = weight(:);
           if norm(weight) > nrgbkdes_params.contrast
              weight = weight/norm(weight);
           else
              weight = weight/nrgbkdes_params.contrast;
           end
           % compute kernel descriptor
           mwkv = ((skv.*(ones(size(skv,1),1)*(weight')))*reshape( im_k(y_lo:y_hi,x_lo:x_hi,:), nrgbkdes_params.patchsize(j)^2, nrgbkdes_params.kdes.rgbsize))';
           mwkvs(:,i) = mwkv(:);
        end
    end
    nrgbkdes_arr = nrgbkdes_params.kdes.eigvectors*mwkvs;
    nrgbkdes_arr(:,it+1:end) = [];
    feaSet.feaArr{j} = nrgbkdes_arr;
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

function I_k = kernel_trans(I, nrgbkdes_params)

[asize,bsize,rgbsize] = size(I);
I_rgb = reshape(I,asize*bsize,rgbsize);
I_k = eval_kernel(I_rgb, nrgbkdes_params.kdes.rgbpoints', nrgbkdes_params.kdes.ktype, nrgbkdes_params.kdes.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,nrgbkdes_params.kdes.rgbsize);


