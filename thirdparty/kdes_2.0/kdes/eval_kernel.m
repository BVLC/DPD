function K = eval_kernel(feaset_1, feaset_2, kernel, kparam)
% evaluate kernel functions
%
%-inputs
% feaset_1       -feature matrix, each row denotes one sample
% feaset_2       -feature matrix, each row denotes one sample
%
%-outputs
% K	         -kernel matrix
% written when Liefeng Bo was in Xidian University
% optimized by Xiaofeng Ren on 2011

if (size(feaset_1,2) ~= size(feaset_2,2))
    error('sample1 and sample2 differ in dimensionality!!');
end
[L1, dim] = size(feaset_1);
[L2, dim] = size(feaset_2);

switch kernel

    % linear kernel
    case 'linear'
        K = feaset_1*feaset_2';

    % polynomial kernel
    case 'poly'
        K = (feaset_1*feaset_2').^kparam;

    % radius basis kernel
    case 'rbf'

        % If sigle parammeter, expand it.
        if length(kparam) < dim
            a = sum(feaset_1.*feaset_1,2);
            b = sum(feaset_2.*feaset_2,2);
            dist2 = bsxfun(@plus, a, b' ) - 2*feaset_1*feaset_2';
            K = exp(-kparam*dist2);
        else
            kparam = kparam(:);
            a = sum(feaset_1.*feaset_1.*repmat(kparam',L1,1),2);
            b = sum(feaset_2.*feaset_2.*repmat(kparam',L2,1),2);
            dist2 = bsxfun(@plus,a,b') - 2*(feaset_1.*repmat(kparam',L1,1))*feaset_2';
            K = exp(-dist2);
        end

    % Laplace kernel
    case 'laplace'
        K = zeros(L1,L2);
        for i = 1:L1
            K(i,:) = sum(abs(bsxfun(@minus, feaset_1(i,:), feaset_2)),2)';
        end
        K = exp(-kparam*K);

    % Euclidean distance
    case 'dist2'
        a = sum(feaset_1.*feaset_1,2);
        b = sum(feaset_2.*feaset_2,2);
        K = bsxfun( @plus, a, b' ) - 2*feaset_1*feaset_2';
            
    % L1 distance
    case 'dist1'
        K = zeros(L1,L2);
        for i = 1:L1
            K(i,:) = sum(abs(bsxfun(@minus, feaset_1(i,:), feaset_2)),2)';
        end
        
    otherwise
        disp('unknown kernel');
end

