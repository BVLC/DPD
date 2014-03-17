function fea_arr = sp_find_sift_grid(I, grid_x, grid_y, patch_size, nrml_threshold, sigma_edge)

% Lazebnik's SIFT code is used.
% rewritten by Liefeng Bo in University of Washington on 27/04/2011
% much more efficient than the original version

% parameters
num_angles = 8;
num_bins = 4;
num_samples = num_bins * num_bins;

if nargin < 6
    sigma_edge = 1;
end

angle_step = 2 * pi / num_angles;
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers

[hgt wid] = size(I);
num_patches = numel(grid_x);

sift_arr = zeros(num_patches, num_samples * num_angles);

[G_X,G_Y] = gen_dgauss(sigma_edge);
%G_X = [-1; 0; 1];
%G_Y = [-1 0 1];

I_X = filter2(G_X, I, 'same'); % vertical edges
I_Y = filter2(G_Y, I, 'same'); % horizontal edges
I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
I_theta = atan2(I_Y,I_X);
I_theta(find(isnan(I_theta))) = 0; % necessary????

% make default grid of samples (centered at zero, width 2)
interval = 2/num_bins:2/num_bins:2;
interval = interval - (1/num_bins + 1);
[sample_x sample_y] = meshgrid(interval, interval);
sample_x = reshape(sample_x, [1 num_samples]);
sample_y = reshape(sample_y, [1 num_samples]);

% make orientation images
I_orientation = zeros(hgt, wid, num_angles);

% for each histogram angle
for a = 1:num_angles    
 
    % compute each orientation channel
    % tmpa = cos(I_theta - angles(a)).^alpha;
    % alpha = 9, speedup;
    tmp = cos(I_theta - angles(a));
    tmp = ((tmp.^2).^2).^2.*tmp;
    tmp = tmp .* (tmp > 0);
    
    % weight by magnitude
    I_orientation(:,:,a) = tmp .* I_mag;
end

for j = 1:length(patch_size)
    
    % find coordinates of sample points (bin centers)
    r = patch_size(j)/2;
    sample_x_t = sample_x * r + r - 0.5;
    sample_y_t = sample_y * r + r - 0.5;
    sample_res = sample_y_t(2) - sample_y_t(1);
    
    % find coordinates of pixels
    dind = (1:patch_size(j)) - 1;
    [sample_px, sample_py] = meshgrid(dind, dind);
    num_pix = numel(sample_px);
    sample_px = reshape(sample_px, [num_pix 1]);
    sample_py = reshape(sample_py, [num_pix 1]);
    
    % find (horiz, vert) distance between each pixel and each grid sample
    dist_px = abs(repmat(sample_px, [1 num_samples]) - repmat(sample_x_t, [num_pix 1]));
    dist_py = abs(repmat(sample_py, [1 num_samples]) - repmat(sample_y_t, [num_pix 1]));
    
    % find weight of contribution of each pixel to each bin
    weights_x = dist_px/sample_res;
    weights_x = (1 - weights_x) .* (weights_x <= 1);
    % weights_x = double(weights_x <= 0.5);
    weights_y = dist_py/sample_res;
    weights_y = (1 - weights_y) .* (weights_y <= 1);
    % weights_y = double(weights_y <= 0.5);
    weights = weights_x .* weights_y;
    
    % for all patches
    for i = 1:num_patches
        
        % find window of pixels that contributes to this descriptor
        x_lo = grid_x(i);
        x_hi = grid_x(i) + patch_size(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + patch_size(j) - 1;
        
        % make sift descriptor
        tmp = reshape(I_orientation(y_lo:y_hi, x_lo:x_hi,:), [num_pix num_angles]);
        curr_sift = tmp'*weights;
        sift_arr(i,:) = reshape(curr_sift, [1 num_samples * num_angles]);
        
    end
    sift_arr = sp_normalize_sift(sift_arr, nrml_threshold);
    fea_arr{j} = sift_arr';
end

function G = gen_gauss(sigma)

if all(size(sigma)==[1, 1])
    % isotropic gaussian
   f_wid = 4 * ceil(sigma) + 1;
   G = fspecial('gaussian', f_wid, sigma);
   
else

    % anisotropic gaussian
    f_wid_x = 2 * ceil(sigma(1)) + 1;
    f_wid_y = 2 * ceil(sigma(2)) + 1;
    G_x = normpdf(-f_wid_x:f_wid_x,0,sigma(1));
    G_y = normpdf(-f_wid_y:f_wid_y,0,sigma(2));
    G = G_y' * G_x;
end

function [GX,GY] = gen_dgauss(sigma)

% laplacian of size sigma

G = gen_gauss(sigma);
[GX,GY] = gradient(G); 

GX = GX * 2 ./ sum(sum(abs(GX)));
GY = GY * 2 ./ sum(sum(abs(GY)));

