% written by Liefeng Bo in University of Washington 02/04/2012

clear;

% add paths
addpath('../kdes');
addpath('../emk');
addpath('../helpfun');
addpath(genpath('../liblinear-1.5-dense-float'));

tic;
% initialize the directories of image
imdir = '../images/bird200/bird200train/';
[trainlabel, impathtrain] = get_im_label(imdir);

% initialize the parameters of kdes
kdes_params.grid = 8;   % kdes is extracted every 8 pixels
kdes_params.patchsize = 16;  % patch size
kdes_params.kdestype{1} = 'gradkdes';
kdes_params.kdestype{2} = 'lbpkdes';
kdes_params.kdestype{3} = 'rgbkdes';
kdes_params.kdestype{4} = 'nrgbkdes';

% initialize the parameters of data
data_params.datapath = impathtrain;
data_params.tag = 1;
data_params.minsize = 45;  % minimum size of image
data_params.maxsize = 300; % maximum size of image
data_params.savedir = '../kdesfeatures/bird200train';

% extract kernel descriptors on training set
savedirtrain = gen_kdes_batch_all(data_params, kdes_params);

% extract kernel descriptors on test set
imdir = '../images/bird200/bird200test/';
[testlabel, impathtest] = get_im_label(imdir);
data_params.datapath = impathtest;
data_params.savedir = '../kdesfeatures/bird200test';
savedirtest = gen_kdes_batch_all(data_params, kdes_params);

% initialize the parameters of basis vectors
basis_params.samplenum = 50; % maximum sample number per image scale
basis_params.wordnum = 1000; % number of visual words
% initialize the parameters of emk
emk_params.pyramid = [1 2 3 4];
emk_params.ktype = 'rbf';
kparam = [0.001, 0.01, 0.01, 0.01];
trainkdes = [];
testkdes = [];
for i = 1:length(kdes_params.kdestype)
  % learn visual words using K-means
  fea_params.feapath = get_kdes_path(savedirtrain{i});
  bird200words = visualwords(fea_params, basis_params);
  % extrac emk features
  basis_params.basis = bird200words;
  emk_params.kparam = kparam(i);
  bird200fea = cksvd_emk_batch(fea_params, basis_params, emk_params);
  trainkdes = [trainkdes; single(bird200fea)];
  fea_params.feapath = get_kdes_path(savedirtest{i});
  bird200fea = cksvd_emk_batch(fea_params, basis_params, emk_params);
  testkdes = [testkdes; single(bird200fea)];
end

% classify with liblinear
disp('Train linear SVM ... ...');
lc = 1; % regularization parameter C
option = ['-s 1 -c ' num2str(lc)];
[trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
testkdes = scaletest(testkdes, 'power', minvalue, maxvalue);
model = train(trainlabel',trainkdes',option);
[predictlabel, accuracy, decvalues] = predict(testlabel', testkdes', model);
bird200_acc_lsvm_allkdes = mean(predictlabel == testlabel');

% print and save classification accuracy
disp(['Accuracy of Liblinear is ' num2str(mean(bird200_acc_lsvm_allkdes))]);
save('./results/bird200_acc_lsvm_allkdes.mat', 'bird200_acc_lsvm_allkdes');


