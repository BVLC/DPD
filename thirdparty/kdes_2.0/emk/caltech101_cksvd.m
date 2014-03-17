
% written by Liefeng Bo in University of Washington on 20/04/2011

clear;

% add paths
addpath('./sift');
addpath('./emk');
addpath('./helpfun');
addpath(genpath('./liblinear-1.51'));

% obtain the paths of images
% don't forget to remove tmp file in BACKGROUND_Google category 
imdir = './images/caltech101/';
caltech101label = get_im_label(imdir);

% extract SIFT
tic;
savedir = './siftfeatures/caltech101sift/';
mkdir_bo(savedir);
grid_space = 6;   % sift is extracted every 8 pixels
patch_size = 24;  % patch size
min_imsize = 45;  % minimum size of image
max_imsize = 300; % maximum size of image
caltech101siftpath = get_sift_path(savedir);
if ~length(caltech101siftpath)
   disp('Extract SIFT descriptors ... ...');
   calculate_sift_descriptor(imdir, savedir, grid_space, patch_size, min_imsize, max_imsize, 'jpg');
   caltech101siftpath = get_sift_path(savedir);
end

% test features with linear SVMs
lc = 10; % regularization parameter C
trail = 2; % results averaged over 5 runs
trainnum = 30; % 30 training images per category
for i = 1:trail

    % generate training and test partitions
    indextrain = [];
    indextest = [];
    labelnum = unique(caltech101label);
    for j = 1:length(labelnum)
        index = find(caltech101label == j);
        perm = randperm(length(index));
        indextrain = [indextrain index(perm(1:trainnum))];
        % training images per category is less than 50
        testnum = min(trainnum+50,length(index)); 
        indextest = [indextest index(perm(trainnum+1:testnum))];
    end

    % learn visual words using K-means
    samplenum = 20; % maximum sample number per image scale
    wordnum = 1000; % number of visual words
    for ss = 1:length(indextrain)
        caltech101siftpathtrain{ss} = caltech101siftpath{indextrain(ss)};
    end
    caltech101words = visualwords_nips(caltech101siftpathtrain, samplenum, wordnum);

    % constrained kernel SVD coding
    disp('Extract image features ... ...');
    pyramid = [1 2 4];
    trainsift = cksvd_emk_batch_nips(caltech101siftpathtrain, caltech101words, pyramid);
    for ss = 1:length(indextest)
        caltech101siftpathtest{ss} = caltech101siftpath{indextest(ss)};
    end
    testsift = cksvd_emk_batch_nips(caltech101siftpathtest, caltech101words, pyramid);

    % generate training and test labels
    trainlabel = caltech101label(:, indextrain);
    testlabel = caltech101label(:, indextest);

    % classify with liblinear
    option = ['-s 2 -c ' num2str(lc)];
    [trainsift, minvalue, maxvalue] = scaletrain(trainsift, 'power');
    testsift = scaletest(testsift, 'power', minvalue, maxvalue);
    trainsift = sparse(trainsift);
    testsift = sparse(testsift);
    model = train(trainlabel',trainsift',option);
    [predictlabel, accuracy, decvalues] = predict(testlabel', testsift', model);
    caltech101_acc_lsvm(i,1) = mean(predictlabel == testlabel');

    % print and save classification accuracy
    disp(['Accuracy of Liblinear is ' num2str(mean(caltech101_acc_lsvm))]);
    save('./results/caltech101_acc_lsvm.mat', 'caltech101_acc_lsvm');
end
totaltime = toc;
disp(['Total running time is ' num2str(totaltime)]);

