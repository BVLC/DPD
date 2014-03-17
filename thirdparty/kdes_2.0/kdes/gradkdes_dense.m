function feaSet = gradkdes_dense(im, gradkdes_params)
% dense gradient kernel descriptors over uniform grids sampled from images/depth maps

% inputs
% im			-images/depth maps
% gradkdes_params	-parameters of gradient kernel descriptors
%
% outputs
% feaSet         	-gradient kernel descriptors and their locations
% written by Liefeng Bo on 2010
% optimized by Xiaofeng Ren 2011
%==========================================================================

if ~isfield(gradkdes_params,'grid'); gradkdes_params.grid = 8; end
if ~isfield(gradkdes_params,'patchsize'); gradkdes_params.patchsize = 16; end
if ~isfield(gradkdes_params,'kdesdim'); gradkdes_params.kdesdim = 200; end
if ~isfield(gradkdes_params,'contrast'); gradkdes_params.contrast = 0.8; end

% default setting
if gradkdes_params.kdesdim ~= 200
   gradkdes_params.kdes.eigvectors = gradkdes_params.kdes.eigvectors(1:gradkdes_params.kdesdim,:);
end

if size(im,3) == 1
   im = im2double(im); % normalize pixel values to [0 1]
else
   im = im2double(rgb2gray(im)); % convert color image to gray image and then normalize pixel values to [0 1]
end

% densely and uniformly sample interest points
mpatchsize = max(gradkdes_params.patchsize); % maximum patch size
[im_h, im_w] = size(im);
rem_x = mod(im_w-mpatchsize, gradkdes_params.grid);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatchsize, gradkdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:gradkdes_params.grid:im_w-mpatchsize+1, offset_y:gradkdes_params.grid:im_h-mpatchsize+1);
num_patches = numel(grid_x);

% HOG-type gradient filters
% G_X = [-1 0 1];
% G_Y = [-1; 0; 1];

sigma_edge = 0.8;
[G_X,G_Y] = gen_dgauss(sigma_edge); % SIFT gradient filters
im_X = filter2(G_X, im, 'same'); % vertical edges
im_Y = filter2(G_Y, im, 'same'); % horizontal edges
im_mag = sqrt(im_X.^2 + im_Y.^2); % gradient magnitude
gvalue = 1e-5; % suppres threshold value
im_mag = max(im_mag, gvalue);
im_o(:,:,1) = im_X./im_mag; % normalized gradient vector
im_o(:,:,2) = im_Y./im_mag; % normalized gradient vector

% gradient basis vectors for all pixels
im_k = kernel_trans(im_o, gradkdes_params);
for j = 1:length(gradkdes_params.patchsize)
    
    xx = repmat((0:gradkdes_params.patchsize(j)-1)/gradkdes_params.patchsize(j), [gradkdes_params.patchsize(j) 1]); % horizontal spatial position
    yy = repmat(((0:gradkdes_params.patchsize(j)-1)')/gradkdes_params.patchsize(j),[1 gradkdes_params.patchsize(j)]); % vertical spatial position

    % spatial basis vectors
    skv = eval_kernel(gradkdes_params.kdes.spoints', [yy(:) xx(:)],gradkdes_params.kdes.ktype,gradkdes_params.kdes.kparam(end-1:end));
    mwkvs = zeros(size(gradkdes_params.kdes.eigvectors,2), num_patches);

    % for all patches
    for i = 1:num_patches
        
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + gradkdes_params.patchsize(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + gradkdes_params.patchsize(j) - 1;
        
        % normalize gradient magnitudes
        weight = im_mag(y_lo:y_hi,x_lo:x_hi);
        weight = weight(:);
        if norm(weight) > gradkdes_params.contrast
            weight = weight/norm(weight);
        else
            weight = weight/gradkdes_params.contrast;
        end
        
        % compute kernel descriptor
        mwkv = ((skv.*(ones(size(skv,1),1)*(weight')))*reshape( im_k(y_lo:y_hi,x_lo:x_hi,:),gradkdes_params.patchsize(j)^2,gradkdes_params.kdes.gsize))';
        mwkvs(:,i) = mwkv(:);
    end
    feaSet.feaArr{j} = gradkdes_params.kdes.eigvectors*mwkvs;
end

% output feature information
feaSet.x = grid_x(:) + mpatchsize/2 - 0.5;
feaSet.y = grid_y(:) + mpatchsize/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function I_k = kernel_trans(I_o, gradkdes_params)
% compute basis vectors

[asize,bsize,gsize] = size(I_o);
I_oo = reshape(I_o,asize*bsize,gsize);
I_k = eval_kernel(I_oo, gradkdes_params.kdes.gpoints', gradkdes_params.kdes.ktype, gradkdes_params.kdes.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,gradkdes_params.kdes.gsize);

function [GX, GY] = gen_dgauss(sigma)

G = gen_gauss(sigma);
[GX,GY] = gradient(G);

GX = GX * 2 ./ sum(sum(abs(GX)));
GY = GY * 2 ./ sum(sum(abs(GY)));

function G = gen_gauss(sigma)

if all(size(sigma) == [1, 1])
    % isotropic gaussian
    f_im_w = 4 * ceil(sigma) + 1;
    G = fspecial('gaussian', f_im_w, sigma);
else
    % anisotropic gaussian
    f_im_w_x = 2 * ceil(sigma(1)) + 1;
    f_im_w_y = 2 * ceil(sigma(2)) + 1;
    G_x = normpdf(-f_im_w_x:f_im_w_x,0,sigma(1));
    G_y = normpdf(-f_im_w_y:f_im_w_y,0,sigma(2));
    G = G_y' * G_x;
end


