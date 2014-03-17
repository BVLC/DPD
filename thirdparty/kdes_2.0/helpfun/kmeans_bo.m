function [centers, sse, group] = kmeans_bo(data, K, maxit)

% written by Liefeng Bo in University of Washington on 27/04/2011

if nargin < 3
   maxit = 50;
end

perm = randperm(size(data,1));
centers = data(perm(1:K),:);
iter = 0;
while iter < maxit
    tic;    
    [centers, sse, group] = kmeans_iter(data, centers);    
    t = toc;
    fprintf('iter %d: sse = %g (%g secs)\n', iter, sse, t)
    iter=iter+1;      
end

function [centers, sse, group] = kmeans_iter(data, centers)

% compute Euclidean distance
a = sum(data.*data,2);
b = sum(centers.*centers,2);
dmatrix = bsxfun( @plus, a, b' ) - 2*data*centers';

% find group index
[ddd, group] = min(dmatrix, [], 2);  % find group index

% compute the loss of kmeans
sse = sum(ddd);

% recompute the centers
for i = 1:size(centers,1)
    ind = find(group == i);
    if length(ind)
        centers(i,:) = mean(data(ind,:),1);
    end
end
