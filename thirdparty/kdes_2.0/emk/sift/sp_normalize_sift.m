function [sift_arr, siftlen] = sp_normalize_sift(sift_arr, threshold)
% normalize SIFT descriptors (after Lowe)
%
% find indices of descriptors to be normalized (those whose norm is larger than 1)
siftlen = sqrt(sum(sift_arr.^2, 2));

normalize_ind1 = find(siftlen > threshold);
normalize_ind2 = find(siftlen <= threshold);

sift_arr_norm1 = sift_arr(normalize_ind1,:);
sift_arr_norm1 = sift_arr_norm1 ./ repmat(siftlen(normalize_ind1,:), [1 size(sift_arr,2)]);

sift_arr_norm2 = sift_arr(normalize_ind2,:);
sift_arr_norm2 = sift_arr_norm2./ threshold;

% suppress large gradients
upper = 0.2;
sift_arr_norm(find(sift_arr_norm1 > upper)) = upper;

% finally, renormalize to unit length
tmp = sqrt(sum(sift_arr_norm1.^2, 2));
sift_arr_norm1 = sift_arr_norm1 ./ repmat(tmp, [1 size(sift_arr,2)]);

sift_arr(normalize_ind1,:) = sift_arr_norm1;
sift_arr(normalize_ind2,:) = sift_arr_norm2;
