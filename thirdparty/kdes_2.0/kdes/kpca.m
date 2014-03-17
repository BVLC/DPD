function [eigvector, eigvalue, centervector, Y] = kpca(data, dim, ktype, kparam)
% Kernel Principal Component Analysis --- KPCA
% written by Liefeng Bo on 2010

% compute the number of datapoints
num = size(data,1);

% compute kernel matrix
K = eval_kernel(data, data, ktype, kparam);
K = max(K, K');
centervector = -mean(K,2) + mean(K(:));

% normalize kernel matrix
K_old = K;
sumK = sum(K, 2);
H = repmat(sumK./num, 1, num);
K = K - H - H' + sum(sumK)/(num^2);
K = max(K, K');
clear H;

if num > 1000 & dim < num/10
    % using eigs to speed up!
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(K,dim,'la',option);
    eigvalue = diag(eigvalue);
else
    [eigvector, eigvalue] = eig(K);
    eigvalue = diag(eigvalue);
    [junk, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:,index);
end

if dim < length(eigvalue)
    eigvector = eigvector(:, 1:dim);
    eigvalue = eigvalue(1:dim);
end

if nargout >= 3
    Y = K_old*eigvector;
end

