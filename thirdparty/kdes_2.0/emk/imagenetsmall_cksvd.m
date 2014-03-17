
% written by Liefeng Bo in University of Washington 01/04/2011

clear;

% add paths
addpath('./sift');
addpath('./emk');
addpath('./helpfun');
addpath(genpath('./liblinear-1.51'));

% obtain the paths of images
imdir = './images/imagenetsmall/';
imagenetlabel = get_im_label(imdir);

% extract kernel descriptors
tic;
savedir = './siftfeatures/imagenetsmallsift/';
mkdir_bo(savedir);
grid_space = 8;   % sift is extracted every 8 pixels
patch_size = 16;  % patch size
min_imsize = 45;  % minimum size of image
max_imsize = 300; % maximum size of image
imagenetsiftpath = get_sift_path(savedir);
if ~length(imagenetsiftpath)
   disp('Extract SIFT descriptors ... ...');
   calculate_sift_descriptor(imdir, savedir, grid_space, patch_size, min_imsize, max_imsize,'jpg');
   imagenetsiftpath = get_sift_path(savedir);
end

% test features with linear SVMs
lc = 10; % regularization parameter C
trail = 5; % results averaged over 5 runs
trainnum = 50; % 50 training images per category
for i = 1:trail

    % generate training and test partitions
    indextrain = [];
    indextest = [];
    labelnum = unique(imagenetlabel);
    for j = 1:length(labelnum)
        index = find(imagenetlabel == j);
        perm = randperm(length(index));
        indextrain = [indextrain index(perm(1:trainnum))];
        indextest = [indextest index(perm(trainnum+1:end))];
    end

    % learn visual words using K-means
    samplenum = 50; % maximum sample number per image scale
    wordnum = 1000; % number of visual words
    for ss = 1:length(indextrain)
        imagenetsiftpathtrain{ss} = imagenetsiftpath{indextrain(ss)};
    end
    imagenetwords = visualwords_nips(imagenetsiftpathtrain, samplenum, wordnum);

    % constrained kernel SVD coding
    disp('Extract image features ... ...');
    pyramid = [1 2 4];
    imagenetfea = cksvd_emk_batch_nips(imagenetsiftpath, imagenetwords, pyramid);

    % generate training and test samples
    trainsift = imagenetfea(:, indextrain);
    trainlabel = imagenetlabel(:, indextrain);
    testsift = imagenetfea(:, indextest);
    testlabel = imagenetlabel(:, indextest);

    % classify with liblinear
    option = ['-s 2 -c ' num2str(lc)];
    [trainsift, minvalue, maxvalue] = scaletrain(trainsift, 'power');
    testsift = scaletest(testsift, 'power', minvalue, maxvalue);
    trainsift = sparse(trainsift);
    testsift = sparse(testsift);
    model = train(trainlabel',trainsift',option);
    [predictlabel, accuracy, decvalues] = predict(testlabel', testsift', model);
    imagenetsmall_acc_lsvm(i,1) = mean(predictlabel == testlabel');

    % print and save classification accuracy
    disp(['Accuracy of Liblinear is ' num2str(mean(imagenetsmall_acc_lsvm))]);
    save('./results/imagenetsmall_acc_lsvm.mat', 'imagenetsmall_acc_lsvm');
end
totaltime = toc;
disp(['Total running time is ' num2str(totaltime)]);

