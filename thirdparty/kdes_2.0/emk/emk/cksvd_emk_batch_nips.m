function [imfea, G] = cksvd_emk_batch_nips(feapath, words, pyramid, ktype, kparam)
% extract image features using constrained kernel SVD
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

if nargin < 4
   ktype = 'rbf';
end
if nargin < 5
   kparam = 1;
end

% set the parameters 
datasize = length(feapath);
wordsnum = size(words,2);
pgrid = pyramid.^2;
sgrid = sum(pgrid);
weights = (1./pgrid); % divided by the number of grids at the coresponding level
weights = weights/sum(weights); 
imfea = zeros(sgrid*wordsnum,datasize);
imlabel = zeros(1,datasize);
K = eval_kernel(words',words',ktype,kparam);
K = K + 1e-6*eye(size(K));
G = chol(inv(K));

% compute the number of the different scale
load(feapath{1});
patchsize = length(feaSet.feaArr);

% each image path
for i = 1:length(feapath)
    load(feapath{i});
    imfea(:,i) = cksvd_emk_nips(feaSet, words, G, pyramid, ktype, kparam);
    if mod(i,10) == 1
       disp(['Current Iteration is: ' num2str(i)]);
    end
end

