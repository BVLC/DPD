
% written by Liefeng Bo in University of Washington 03/26/2012

clear;

% add paths
addpath('../kdes');
addpath('../emk');
addpath('../lsvm');
addpath('../helpfun');

% obtain the paths of images
imdir = '../images/imagenetsmall/';
[imagenetsmalllabel, impath] = get_im_label(imdir);

% extract kernel descriptors
tic;

% initialize the parameters of kdes
kdes_params.grid = 8;   % kdes is extracted every 8 pixels
kdes_params.patchsize = 16;  % patch size
load('lbpkdes_params');
kdes_params.kdes = lbpkdes_params;

% initialize the parameters of data
data_params.datapath = impath;
data_params.tag = 1;
data_params.minsize = 45;  % minimum size of image
data_params.maxsize = 300; % maximum size of image
data_params.savedir = ['../kdesfeatures/imagenetsmall' 'lbpkdes'];

mkdir_bo(data_params.savedir);
imagenetsmallkdespath = get_kdes_path(data_params.savedir);
if ~length(imagenetsmallkdespath)
   gen_kdes_batch(data_params, kdes_params);
   imagenetsmallkdespath = get_kdes_path(data_params.savedir);
end

% test features with linear SVMs
lc = 10; % regularization parameter C
trail = 10; % results averaged over 5 runs
trainnum = 50; % 50 training images per category
for i = 1:trail

    % generate training and test partitions
    indextrain = [];
    indextest = [];
    labelnum = unique(imagenetsmalllabel);
    for j = 1:length(labelnum)
        index = find(imagenetsmalllabel == j);
        perm = randperm(length(index));
        indextrain = [indextrain index(perm(1:trainnum))];
        indextest = [indextest index(perm(trainnum+1:end))];
    end

    % learn visual words using K-means
    % initialize the parameters of basis vectors
    basis_params.samplenum = 50; % maximum sample number per image scale
    basis_params.wordnum = 1000; % number of visual words
    for ss = 1:length(indextrain)
        imagenetsmallkdespathtrain{ss} = imagenetsmallkdespath{indextrain(ss)};
    end
    fea_params.feapath = imagenetsmallkdespathtrain;
    imagenetsmallwords = visualwords(fea_params, basis_params);
    basis_params.basis = imagenetsmallwords;

    % constrained kernel SVD coding
    disp('Extract image features ... ...');
    % initialize the params of emk
    emk_params.pyramid = [1 2 4];
    emk_params.ktype = 'rbf';
    emk_params.kparam = 0.01;
    fea_params.feapath = imagenetsmallkdespath;
    imagenetsmallfea = cksvd_emk_batch(fea_params, basis_params, emk_params);

    % generate training and test samples
    trainkdes = imagenetsmallfea(:, indextrain);
    trainlabel = imagenetsmalllabel(:, indextrain);
    testkdes = imagenetsmallfea(:, indextest);
    testlabel = imagenetsmalllabel(:, indextest);

    % classify with linear SVM
    disp('Train linear SVM ... ...');
    [trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
    testkdes = scaletest(testkdes, 'power', minvalue, maxvalue);
    lambda = 1/2/lc;
    model = lsvm_train(trainkdes', trainlabel', lambda);
    predictlabel = lsvm_predict(testkdes', model);
    imagenetsmall_acc_lsvm_lbpkdes(i,1) = mean(predictlabel == testlabel');

    % print and save classification accuracy
    disp(['Accuracy of Liblinear is ' num2str(mean(imagenetsmall_acc_lsvm_lbpkdes))]);
    save('./results/imagenetsmall_acc_lsvm_lbpkdes.mat', 'imagenetsmall_acc_lsvm_lbpkdes');
end
totaltime = toc;
disp(['Total running time is ' num2str(totaltime)]);


