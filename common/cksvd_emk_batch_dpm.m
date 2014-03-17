function [imfea,G] = cksvd_emk_batch_dpm(fea_params, basis_params, emk_params, pooling_params)
% extract image features using constrained kernel SVD
% written by Ning Zhang, 03/07/2013

% The setup of basis_params.G has moved higher up in the call stack
%K = eval_kernel(basis_params.basis', basis_params.basis', emk_params.ktype, emk_params.kparam); %TODO: precompute in dpd_set_up.m, or store in a .mat file
%K = K + 1e-6*eye(size(K));
%G = chol(inv(K));
%basis_params.G = G;
G = basis_params.G; %does anyone ever use this as a return value?

% compute the number of the different scale
% load(fea_params.feapath{1});
% patchsize = length(feaSet.feaArr);
%imfea = zeros(210000,length(fea_params.feapath));
% each image path

%if we're in batch mode (lots of image in feapath), then compute the 1st
% image separately to determine the size when preallocating 'imfea'
if(length(fea_params.feapath) > 1)
    load(fea_params.feapath{1}); %load feaSet (KDES feature descriptor)
    params.feaSet = feaSet;
    bbox = pooling_params.bb(1,:);
    for t =1:numel(pooling_params.parts)
        parts{t} = pooling_params.parts{t}(1,:);
    end
    imfea1 = cksvd_emk_parts(params, basis_params, emk_params, bbox,parts);
    imfea =zeros(length(imfea1) ,length(fea_params.feapath));
end

for i = 1:length(fea_params.feapath)
    load(fea_params.feapath{i}); %load feaSet (KDES feature descriptor)
    params.feaSet = feaSet;
    bbox = pooling_params.bb(i,:);
    for t =1:numel(pooling_params.parts)
        parts{t} = pooling_params.parts{t}(i,:);
    end
    imfea(:,i)= cksvd_emk_parts(params, basis_params, emk_params, bbox,parts);
    if mod(i,10) == 1
        disp(['Current Iteration is: ' num2str(i)]);
    end
    
end

%TODO: avoid having the same code written twice in here


