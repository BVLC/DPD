% written by Liefeng Bo in University of Washington 02/04/2012

clear;

% add paths
addpath('../kdes');
addpath('../emk');
addpath(genpath('../liblinear-1.5-dense'));
addpath('../helpfun');

tic;
% initialize the directories of image
imdir = '../images/usps/uspstrain/';
[trainlabel, impathtrain] = get_im_label(imdir);

% initialize the parameters of kdes
kdes_params.grid = 1;   % kdes is extracted every 8 pixels
kdes_params.patchsize = 6;  % patch size
kdes_params.kdestype{1} = 'gradkdes';
kdes_params.kdestype{2} = 'lbpkdes';
kdes_params.kdestype{3} = 'rgbkdes';

% initialize the parameters of data
data_params.datapath = impathtrain;
data_params.tag = 0;
data_params.minsize = 1;  % minimum size of image
data_params.maxsize = 300; % maximum size of image
data_params.savedir = '../kdesfeatures/uspstrain';

% extract kernel descriptors on training set
savedirtrain = gen_kdes_batch_all(data_params, kdes_params);

% extract kernel descriptors on test set
imdir = '../images/usps/uspstest/';
[testlabel, impathtest] = get_im_label(imdir);
data_params.datapath = impathtest;
data_params.savedir = '../kdesfeatures/uspstest';
savedirtest = gen_kdes_batch_all(data_params, kdes_params);

% initialize the parameters of basis vectors
basis_params.samplenum = 30; % maximum sample number per image scale
basis_params.wordnum = 400; % number of visual words
% initialize the parameters of emk
emk_params.pyramid = [1 2 4];
emk_params.ktype = 'rbf';
kparam = [0.001, 0.01, 0.01];
trainkdes = [];
testkdes = [];
for i = 1:length(kdes_params.kdestype)
  % learn visual words using K-means
  fea_params.feapath = get_kdes_path(savedirtrain{i});
  uspswords = visualwords(fea_params, basis_params);
  % extrac emk features
  basis_params.basis = uspswords;
  emk_params.kparam = kparam(i);
  uspsfea = cksvd_emk_batch(fea_params, basis_params, emk_params);
  trainkdes = [trainkdes; uspsfea];
  fea_params.feapath = get_kdes_path(savedirtest{i});
  uspsfea = cksvd_emk_batch(fea_params, basis_params, emk_params);
  testkdes = [testkdes; uspsfea];
end

% classify with liblinear
disp('Train linear SVM ... ...');
lc = 1; % regularization parameter C
option = ['-s 1 -c ' num2str(lc)];
[trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
testkdes = scaletest(testkdes, 'power', minvalue, maxvalue);
model = train(trainlabel',trainkdes',option);
[predictlabel, accuracy, decvalues] = predict(testlabel', testkdes', model);
usps_acc_lsvm_allkdes = mean(predictlabel == testlabel');

% print and save classification accuracy
disp(['Accuracy of Liblinear is ' num2str(mean(usps_acc_lsvm_allkdes))]);
save('./results/usps_acc_lsvm_allkdes.mat', 'usps_acc_lsvm_allkdes');


