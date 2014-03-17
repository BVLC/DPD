% written by Liefeng Bo in University of Washington 02/04/2012

clear;

% add paths
addpath('../kdes');
addpath('../emk');
addpath('../helpfun');
addpath('../lsvm');

tic;
% initialize the directories of image
imdir = '../images/imagenetsmall/';
[imagenetsmalllabel, impath] = get_im_label(imdir);

% initialize the parameters of kdes
kdes_params.grid = 8;   % kdes is extracted every 8 pixels
kdes_params.patchsize = 16;  % patch size
kdes_params.kdestype{1} = 'gradkdes';
kdes_params.kdestype{2} = 'lbpkdes';
kdes_params.kdestype{3} = 'rgbkdes';

% initialize the parameters of data
data_params.datapath = impath;
data_params.tag = 1;
data_params.minsize = 45;  % minimum size of image
data_params.maxsize = 300; % maximum size of image
data_params.savedir = '../kdesfeatures/imagenetsmall';

% extract kernel descriptors on training set
savedirtrain = gen_kdes_batch_all(data_params, kdes_params);

% initialize the parameters of basis vectors
basis_params.samplenum = 50; % maximum sample number per image scale
basis_params.wordnum = 1000; % number of visual words
% initialize the parameters of emk
emk_params.pyramid = [1 2 4];
emk_params.ktype = 'rbf';
kparam = [0.001, 0.01, 0.01];
imagenetsmallfea_all = [];
for i = 1:length(kdes_params.kdestype)
  % learn visual words using K-means
  fea_params.feapath = get_kdes_path(savedirtrain{i});
  imagenetsmallwords = visualwords(fea_params, basis_params);
  % extrac emk features
  basis_params.basis = imagenetsmallwords;
  emk_params.kparam = kparam(i);
  imagenetsmallfea = cksvd_emk_batch(fea_params, basis_params, emk_params);
  imagenetsmallfea_all = [imagenetsmallfea_all; single(imagenetsmallfea)];
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

    % generate training and test samples
    trainkdes = imagenetsmallfea_all(:, indextrain);
    trainlabel = imagenetsmalllabel(:, indextrain);
    testkdes = imagenetsmallfea_all(:, indextest);
    testlabel = imagenetsmalllabel(:, indextest);

    % classify with linear SVM
    disp('Train linear SVM ... ...');
    [trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
    testkdes = scaletest(testkdes, 'power', minvalue, maxvalue);
    lambda = 1/2/lc;
    model = lsvm_train(trainkdes', trainlabel', lambda);
    predictlabel = lsvm_predict(testkdes', model);
    imagenetsmall_acc_lsvm_allkdes(i,1) = mean(predictlabel == testlabel');

    % print and save classification accuracy
    disp(['Accuracy of Liblinear is ' num2str(mean(imagenetsmall_acc_lsvm_allkdes))]);
    save('./results/imagenetsmall_acc_lsvm_allkdes.mat', 'imagenetsmall_acc_lsvm_allkdes');
end
totaltime = toc;
disp(['Total running time is ' num2str(totaltime)]);


