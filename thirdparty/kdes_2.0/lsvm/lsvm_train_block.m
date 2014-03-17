function model = lsvm_train_block(trainpath, trainlabel, lambda, maxit)
% train multi-class SVMs with one-vs-all coding on large data that cannot fit in memory
% written by Liefeng Bo on 01/04/2011 in University of Washington

if nargin < 4
   maxit = 10;
end

% set parameters
global X;
opt.lin_cg = 0;
% opt.cg = 1;

% get labels
ulabel = unique(trainlabel);

% initialize the biases and decision functions
b = zeros(length(trainpath),length(ulabel));
pred_old = zeros(length(trainlabel),length(ulabel));
for i = 1:maxit
    start = 0;
    for j = 1:length(trainpath)
        % load training features
        X = load(trainpath{j});
        blockindex = start + (1:size(X,2));

        for s = 1:length(ulabel)
            % compute one-vs-all codes
            binarylabel = trainlabel;
            binarylabel(binarylabel ~= s) = -1;
            binarylabel(binarylabel == s) = 1;

            % update the decision functions 
            if i ~= 1
               pred_old(:,s) = pred_old(:,s) - X*w(blockindex,s) - b(j,s);
            end

            % train binary linear SVM with finite Newton methods
            [w_block, b_block] = primal_svm_block(1, binarylabel, lambda, pred_old(:,s), opt);

            % update the decision functions
            pred_old(:,s) = pred_old(:,s) + X*w_block + b_block;
            w(blockindex,s) = w_block;
            b(j,s) = b_block;

            % print training information
            out = 1 - binarylabel.*pred_old(:,s);
            out = max(0,out);
            obj((i-1)*length(trainpath)+j,s) = sum(out.^2)/2 + lambda*w(:,s)'*w(:,s)/2;
            disp(['Squared hinge loss of ' num2str(s) 'th binary classifier = ' num2str(obj(end,s))]);
        end

        start = start + size(X,2);

    end
    disp(['Overall iteration is ' num2str(i)]);
end

% save learned SVM models
model.ulabel = ulabel;
model.w = w;
model.b = sum(b,1);
model.obj = obj;


