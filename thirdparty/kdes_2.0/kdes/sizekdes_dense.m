function feaSet = sizekdes_dense(pcloud, sizekdes_params)
%% dense size kernel descriptors over uniform grids sampled from point cloud
% inputs
% pcloud                -point clouds
% sizekdes_params       -parameters of size kernel descriptors
%
% outputs
% feaSet                -size kernel descriptors and their locations
% modified by Liefeng Bo on 2012
% optimized by Xiaofeng Ren on 2012
%==========================================================================

if ~isfield(sizekdes_params,'grid'); sizekdes_params.grid = 8; end
if ~isfield(sizekdes_params,'patchsize'); sizekdes_params.patchsize = 16; end
if ~isfield(sizekdes_params,'kdesdim'); sizekdes_params.kdesdim = 50; end

% default setting
if sizekdes_params.kdesdim ~= 50
   sizekdes_params.kdes.eigvectors = sizekdes_params.kdes.eigvectors(1:sizekdes_params.kdesdim,:);
end

% densely and uniformly sample interest points
mpatchsize = max(sizekdes_params.patchsize); % maximum patch size
mpatchsize_half = round(mpatchsize/2); % half maximal patch size

[cloud_h, cloud_w, cdim] = size(pcloud);
rem_x = mod(cloud_w-mpatchsize_half, sizekdes_params.grid);
offset_x = floor(rem_x/2)+1;
rem_y = mod(cloud_h-mpatchsize_half, sizekdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:sizekdes_params.grid:cloud_w-mpatchsize_half+1, offset_y:sizekdes_params.grid:cloud_h-mpatchsize_half+1);
num_patches = numel(grid_x);

% size basis vector for all pixels
pcloudvector = reshape(pcloud, size(pcloud,1)*size(pcloud,2),size(pcloud,3));
ind = find(pcloudvector(:,3) > 0);
sizekdes_arr = zeros(size(sizekdes_params.kdes.eigvectors,1), num_patches);
% for all patches
it = 0;
keep = [];
for i = 1:num_patches
        
    % coordinates of key points
    rpoint(1,1) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 1);
    rpoint(1,2) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 2);
    rpoint(1,3) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 3);
    if rpoint(3) > 0
       it = it + 1;
       keep = [keep i];
       % randomly sample 3d points
       perm = randsample(length(ind),min(length(ind), sizekdes_params.patchsize^2),0);
       subpcloud = pcloudvector(ind(perm),:);
        
       % compute size kernel descriptor
       dist = sqrt(sum((subpcloud - repmat(rpoint,size(subpcloud,1),1)).^2,2));
       kdist = eval_kernel(sizekdes_params.kdes.points', dist, sizekdes_params.kdes.ktype, sizekdes_params.kdes.kparam);
       mkdist = mean(kdist,2);
       sizekdes_arr(:,it) = (sizekdes_params.kdes.eigvectors*mkdist(:));
    end
end

% output size kernel descriptors
sizekdes_arr(:,it+1:end) = [];
feaSet.feaArr{1} = sizekdes_arr;
feaSet.x = grid_x(keep) + mpatchsize_half- 0.5;
feaSet.y = grid_y(keep) + mpatchsize_half- 0.5;
feaSet.width = cloud_w;
feaSet.height = cloud_h;

