function feaSet = lbpkdes_dense(im, lbpkdes_params)
% dense local binary pattern kernel descriptors over uniform grids sampled from images/depth maps

% inputs
% im             -images/depth maps
% lbpkdes_params -parameters of local binary pattern kernel descriptors
%
% outputs
% feaSet         -local binary pattern kernel descriptors and their locations
% modified by Liefeng Bo on March 26, 2012
%==========================================================================

if ~isfield(lbpkdes_params,'grid'); lbpkdes_params.grid = 8; end
if ~isfield(lbpkdes_params,'patchsize'); lbpkdes_params.patchsize = 16; end
if ~isfield(lbpkdes_params,'kdesdim'); lbpkdes_params.kdesdim = 200; end
if ~isfield(lbpkdes_params,'contrast'); lbpkdes_params.contrast = 0.8; end

% default setting
if lbpkdes_params.kdesdim ~= 200
   lbpkdes_params.kdes.eigvectors = lbpkdes_params.kdes.eigvectors(1:lbpkdes_params.kdesdim,:);
end

if size(im,3) == 1
    im = im2double(im); % normalize pixel values to [0 1]
else
    im = im2double(rgb2gray(im)); % convert color image to gray image and then normalize pixel values to [0 1]
end

% densely and uniformly sample interest points
mpatchsize = max(lbpkdes_params.patchsize); % maximum patch size
[im_h, im_w] = size(im);
rem_x = mod(im_w-mpatchsize, lbpkdes_params.grid);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatchsize, lbpkdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:lbpkdes_params.grid:im_w-mpatchsize+1, offset_y:lbpkdes_params.grid:im_h-mpatchsize+1);
num_patches = numel(grid_x);

% local binary pattern
im_c = lbp_trans(im);
% local binary pattern basis vectors for all pixels
im_k = kernel_trans(im_c, lbpkdes_params);

% for all patches
for j = 1:length(lbpkdes_params.patchsize)
    xx = repmat((0:lbpkdes_params.patchsize(j)-1)/lbpkdes_params.patchsize(j), [lbpkdes_params.patchsize(j) 1]); % horizontal spatial position
    yy = repmat(((0:lbpkdes_params.patchsize(j)-1)')/lbpkdes_params.patchsize(j),[1 lbpkdes_params.patchsize(j)]); % vertical spatial position
    
    % spatial basis vectors
    skv = eval_kernel(lbpkdes_params.kdes.spoints', [yy(:) xx(:)],lbpkdes_params.kdes.ktype,lbpkdes_params.kdes.kparam(end-1:end));
    mwkvs = zeros( size(lbpkdes_params.kdes.eigvectors,2), num_patches);
    
    for i = 1:num_patches
 
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + lbpkdes_params.patchsize(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + lbpkdes_params.patchsize(j) - 1;
        
        % normalize standard deviations
        weight = im_c(y_lo:y_hi,x_lo:x_hi,end);
        weight = weight(:);
        if norm(weight) > lbpkdes_params.contrast
            weight = weight/norm(weight);
        else
            weight = weight/lbpkdes_params.contrast;
        end
        
        % compute kernel descriptor
        mwkv = ( (skv.*(ones(size(skv,1),1)*(weight')))*reshape(im_k(y_lo:y_hi,x_lo:x_hi,:), lbpkdes_params.patchsize(j)^2, lbpkdes_params.kdes.lbpsize) )';
        mwkvs(:,i) = mwkv(:);
    end
    feaSet.feaArr{j} = lbpkdes_params.kdes.eigvectors*mwkvs;
end

% output feature information
feaSet.x = grid_x(:) + mpatchsize/2 - 0.5;
feaSet.y = grid_y(:) + mpatchsize/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function I_k = kernel_trans(I_c, lbpkdes_params)

[asize,bsize,csize] = size(I_c);
I_cc = reshape(I_c,asize*bsize,csize);
I_k = eval_kernel(I_cc(:,1:end-1), lbpkdes_params.kdes.lbppoints', lbpkdes_params.kdes.ktype, lbpkdes_params.kdes.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,lbpkdes_params.kdes.lbpsize);


