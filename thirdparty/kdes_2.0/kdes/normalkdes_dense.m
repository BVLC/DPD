function feaSet = normalkdes_dense(pcloud, normalkdes_params)
%% dense normal kernel descriptors over uniform grids sampled from 3D point cloud
% inputs
% pcloud		  -point clouds
% normalkdes_params       -parameters of normal kernel descriptors
%
% outputs
% feaSet		  -normal kernel descriptors and their locations
% written by Liefeng Bo on 2012
% optimized by Xiaofeng Ren on 2012
%==========================================================================

if ~isfield(normalkdes_params,'grid'); normalkdes_params.grid = 8; end
if ~isfield(normalkdes_params,'patchsize'); normalkdes_params.patchsize = 40; end
if ~isfield(normalkdes_params,'kdesdim'); normalkdes_params.kdesdim = 200; end
if ~isfield(normalkdes_params,'radius'); normalkdes_params.radius = 0.05; end

% default setting
if normalkdes_params.kdesdim ~= 200
   normalkdes_params.kdes.eigvectors = normalkdes_params.kdes.eigvectors(1:normalkdes_params.kdesdim,:);
end
maxsample = 256;
minsample = 10;

% densely and uniformly sample interest points
mpatchsize = max(normalkdes_params.patchsize); % maximal patch size
mpatchsize_half = round(mpatchsize/2); % half maximal patch size
[cloud_h, cloud_w, cdim] = size(pcloud);
rem_x = mod(cloud_w - mpatchsize_half, normalkdes_params.grid);    % use half mpatchsize for boundary margin
offset_x = floor(rem_x/2)+1;
rem_y = mod(cloud_h - mpatchsize_half, normalkdes_params.grid);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:normalkdes_params.grid:cloud_w-mpatchsize_half+1, offset_y:normalkdes_params.grid:cloud_h-mpatchsize_half+1);
num_patches = numel(grid_x);

% normal basis vector for all pixels
normal = pcnormal(pcloud);
pcloudvector = reshape(pcloud, size(pcloud,1)*size(pcloud,2), size(pcloud,3));
normalkdes_arr = zeros(size(normalkdes_params.kdes.eigvectors,1), num_patches);
% for all patches
it = 0;
keep = [];

mwkvs=zeros( size(normalkdes_params.kdes.eigvectors,2),num_patches);

for i = 1:num_patches
        
    % coordinates and normals of key points
    rpoint(1,1) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 1);
    rpoint(1,2) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 2);
    rpoint(1,3) = pcloud(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 3);
    cnormal(1,1) = normal(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 1);
    cnormal(1,2) = normal(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 2);
    cnormal(1,3) = normal(grid_y(i)+mpatchsize_half-1, grid_x(i)+mpatchsize_half-1, 3);
    if rpoint(3) > 0

       % search region
       minh = grid_y(i);
       maxh = min(grid_y(i) + normalkdes_params.patchsize, size(pcloud,1));
       minw = grid_x(i);
       maxw = min(grid_x(i) + normalkdes_params.patchsize, size(pcloud,2));

       % sample nearest points and their normals
       subpcloud = pcloud(minh:maxh, minw:maxw,:);
       subnormal = normal(minh:maxh, minw:maxw,:);
       subpcloud = reshape(subpcloud, size(subpcloud,1)*size(subpcloud,2), 3);
       subnormal = reshape(subnormal, size(subnormal,1)*size(subnormal,2), 3);
       diff = subpcloud - repmat(rpoint, size(subpcloud,1), 1);
       dist = sqrt(sum(diff.^2, 2));
       index = find(dist < normalkdes_params.radius & subpcloud(:,3) > 0);
       if length(index) > minsample
          it = it + 1;
          keep = [keep i];
          perm = randsample(length(index),min(maxsample,length(index)),0);
          subpcloud = subpcloud(index(perm),:);
          subnormal = subnormal(index(perm),:);
          subnormal = subnormal./(repmat(sqrt(sum(subnormal.^2,2)),1,size(subnormal,2))+eps);   % unnecessary

          % compute spin-type distance and angles of normals
          cnormalvector = repmat(cnormal,size(subpcloud,1),1);
          normal_x = sum(cnormalvector.*diff(index(perm),:), 2);
          normal_y_square = dist(index(perm)).^2 - normal_x.^2;
          normal_y_square(normal_y_square < 0) = 0;
          normal_y = sqrt(normal_y_square);
          nndot = sum(cnormalvector.*subnormal,2);
          nndot(nndot > 1) = 1 - eps;
          nndot(nndot < -1) = -1 + eps;
          angle = acos(nndot);

          % compute normal kernel descriptor
          nkv = eval_kernel(normalkdes_params.kdes.npoints', [sin(angle) cos(angle)], normalkdes_params.kdes.ktype, normalkdes_params.kdes.kparam(1:2));
          normalkv = eval_kernel(normalkdes_params.kdes.spoints', [normal_x(:) normal_y(:)], normalkdes_params.kdes.ktype, normalkdes_params.kdes.kparam(3:4));
          mwkv = (nkv*normalkv')/size(nkv,2);
          mwkvs(:,it) = mwkv(:);
      end
   end
end
normalkdes_arr = normalkdes_params.kdes.eigvectors*mwkvs(:,1:it);

% output normal kernel descriptors
feaSet.feaArr{1} = normalkdes_arr;
feaSet.x = grid_x(keep) + mpatchsize_half - 0.5;
feaSet.y = grid_y(keep) + mpatchsize_half - 0.5;
feaSet.width = cloud_w;
feaSet.height = cloud_h;

