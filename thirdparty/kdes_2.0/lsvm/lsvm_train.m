function model = lsvm_train(trainfea, trainlabel, lambda)
% train multi-class SVMs with one-vs-all coding
% written by Liefeng Bo on 01/04/2011 in University of Washington

% set parameters
global X; X = trainfea;
opt.lin_cg = 1;
% opt.cg = 1;

% get labels
ulabel = unique(trainlabel);

% initialize the weights and biases
w = zeros(size(trainfea,2),length(ulabel));
b = zeros(1,length(ulabel));
for i = 1:length(ulabel)

    % compute one-vs-all codes
    binarylabel = trainlabel;
    binarylabel(binarylabel ~= i) = -1;
    binarylabel(binarylabel == i) = 1;

    % train binary linear SVM with finite Newton methods
    [w(:,i), b(:,i)] = primal_svm(1, binarylabel, lambda, opt);

    % print training information
    disp(['ID of binary classifier is ' num2str(i)]);
end

% save learned SVM models
model.ulabel = ulabel;
model.w = w;
model.b = b;

