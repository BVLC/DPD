function [imfea, G] = cksvd_emk_batch(fea_params, basis_params, emk_params)
% extract image features using constrained kernel SVD
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

% set the parameters 
datasize = length(fea_params.feapath);
basisnum = size(basis_params.basis,2);
pgrid = emk_params.pyramid.^2;
sgrid = sum(pgrid);
weights = (1./pgrid); % divided by the number of grids at the coresponding level
weights = weights/sum(weights); 
imfea = zeros(sgrid*basisnum,datasize);
imlabel = zeros(1,datasize);
K = eval_kernel(basis_params.basis', basis_params.basis', emk_params.ktype, emk_params.kparam);
K = K + 1e-6*eye(size(K));
G = chol(inv(K));
basis_params.G = G;

% compute the number of the different scale
load(fea_params.feapath{1});
patchsize = length(feaSet.feaArr);

% each image path
for i = 1:length(fea_params.feapath)
    load(fea_params.feapath{i});
    fea_params.feaSet = feaSet;
    imfea(:,i) = cksvd_emk(fea_params, basis_params, emk_params);
    if mod(i,10) == 1
       disp(['Current Iteration is: ' num2str(i)]);
    end
end

