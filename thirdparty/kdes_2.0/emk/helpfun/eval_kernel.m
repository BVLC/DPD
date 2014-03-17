function K = eval_kernel(samples1, samples2, kernel, kernelparam)
% evaluate kernel functions
% mainly written when Liefeng Bo was in Xidian University

if (size(samples1,2)~=size(samples2,2))
    error('sample1 and sample2 differ in dimensionality!!');
end
[L1, dim] = size(samples1);
[L2, dim] = size(samples2);

switch kernel

    % linear kernel
    case 'linear'
        K = samples1*samples2';

    % polynomial kernel
    case 'poly'
        K = (samples1*samples2').^kernelparam;

    % radius basis kernel
    case 'rbf'

        % If sigle parammeter, expand it.
        if length(kernelparam) < dim
            a = sum(samples1.*samples1,2);
            b = sum(samples2.*samples2,2);
            dist2 = bsxfun( @plus, a, b' ) - 2*samples1*samples2';
            K = exp(-kernelparam*dist2);
        else
            kernelparam = kernelparam(:);
            a = sum(samples1.*samples1.*repmat(kernelparam',L1,1),2);
            b = sum(samples2.*samples2.*repmat(kernelparam',L2,1),2);
            dist2 = bsxfun(@plus,a,b') - 2*(samples1.*repmat(kernelparam',L1,1))*samples2';
            K = exp(-dist2);
        end

    % Laplace kernel
    case 'laplace'
        K = zeros(L1,L2);
        for i = 1:L1
            K(i,:) = sum(abs(bsxfun(@minus, samples1(i,:), samples2)),2)';
        end
        K = exp(-kernelparam*K);

    % Euclidean distance
    case 'dist2'
        a = sum(samples1.*samples1,2);
        b = sum(samples2.*samples2,2);
        K = bsxfun( @plus, a, b' ) - 2*samples1*samples2';
            
    % L1 distance
    case 'dist1'
        K = zeros(L1,L2);
        for i = 1:L1
            K(i,:) = sum(abs(bsxfun(@minus, samples1(i,:), samples2)),2)';
        end
        
    otherwise
        disp('unknown kernel');
end

