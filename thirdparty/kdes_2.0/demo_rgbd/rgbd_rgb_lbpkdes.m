
% written by Liefeng Bo on 03/27/2012 in University of Washington

clear;

% add paths
addpath('../liblinear-1.5-dense-float/matlab');
addpath('../helpfun');
addpath('../kdes');
addpath('../emk');

% compute the paths of images
imdir = '../images/rgbdsubset/';
imsubdir = dir_bo(imdir);
impath = [];
rgbdclabel = [];
rgbdilabel = [];
rgbdvlabel = [];
subsample = 1;
label_num = 0;
for i = 1:length(imsubdir)
    [rgbdilabel_tmp, impath_tmp] = get_im_label([imdir imsubdir(i).name '/'], '_crop.png');
    for j = 1:length(impath_tmp)
        ind = find(impath_tmp{j} == '_');
        rgbdvlabel_tmp(1,j) = str2num(impath_tmp{j}(ind(end-2)+1));
    end

    it = 0;
    for j = 1:subsample:length(impath_tmp)
        it = it + 1;
        impath_tmp_sub{it} = impath_tmp{j};
    end
    impath = [impath impath_tmp_sub];
    rgbdclabel = [rgbdclabel i*ones(1,length(impath_tmp_sub))];
    rgbdilabel = [rgbdilabel rgbdilabel_tmp(1:subsample:end)+label_num];
    rgbdvlabel = [rgbdvlabel rgbdvlabel_tmp(1:subsample:end)];
    label_num = label_num + length(unique(rgbdilabel_tmp));
    clear impath_tmp_sub rgbdvlabel_tmp;
end

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
data_params.savedir = ['../kdesfeatures/rgbd' 'lbpkdes'];

% extract kernel descriptors
mkdir_bo(data_params.savedir);
rgbdkdespath = get_kdes_path(data_params.savedir);
if ~length(rgbdkdespath)
   gen_kdes_batch(data_params, kdes_params);
   rgbdkdespath = get_kdes_path(data_params.savedir);
end

featag = 1;
if featag
   % learn visual words using K-means
   % initialize the parameters of basis vectors
   basis_params.samplenum = 10; % maximum sample number per image scale
   basis_params.wordnum = 1000; % number of visual words
   fea_params.feapath = rgbdkdespath;
   rgbdwords = visualwords(fea_params, basis_params);
   basis_params.basis = rgbdwords;

   % constrained kernel SVD coding
   disp('Extract image features ... ...');
   % initialize the params of emk
   emk_params.pyramid = [1 2 3];
   emk_params.ktype = 'rbf';
   emk_params.kparam = 0.01;
   fea_params.feapath = rgbdkdespath;
   rgbdfea = cksvd_emk_batch(fea_params, basis_params, emk_params);
   rgbdfea = single(rgbdfea);
   save -v7.3 rgbdfea_rgb_lbpkdes rgbdfea rgbdclabel rgbdilabel rgbdvlabel;
else
   load rgbdfea_rgb_lbpkdes;
end

category = 1;
if category
   trail = 5;
   for i = 1:trail
       % generate training and test samples
       ttrainindex = [];
       ttestindex = [];
       labelnum = unique(rgbdclabel);
       for j = 1:length(labelnum)
           trainindex = find(rgbdclabel == labelnum(j));
           rgbdilabel_unique = unique(rgbdilabel(trainindex));
           perm = randperm(length(rgbdilabel_unique));
           subindex = find(rgbdilabel(trainindex) == rgbdilabel_unique(perm(1)));
           testindex = trainindex(subindex);
           trainindex(subindex) = [];
           ttrainindex = [ttrainindex trainindex];
           ttestindex = [ttestindex testindex];
       end
       load rgbdfea_rgb_lbpkdes;
       trainhmp = rgbdfea(:,ttrainindex);
       clear rgbdfea;
       [trainhmp, minvalue, maxvalue] = scaletrain(trainhmp, 'power');
       trainlabel = rgbdclabel(ttrainindex); % take category label

       % classify with liblinear
       lc = 10;
       option = ['-s 1 -c ' num2str(lc)];
       model = train(trainlabel',trainhmp',option);
       load rgbdfea_rgb_lbpkdes;
       testhmp = rgbdfea(:,ttestindex);
       clear rgbdfea;
       testhmp = scaletest(testhmp, 'power', minvalue, maxvalue);
       testlabel = rgbdclabel(ttestindex); % take category label
       [predictlabel, accuracy, decvalues] = predict(testlabel', testhmp', model);
       acc_c(i,1) = mean(predictlabel == testlabel');
       save('./results/rgb_lbpkdes_acc_c.mat', 'acc_c', 'predictlabel', 'testlabel');

       % print and save results
       disp(['Accuracy of Liblinear is ' num2str(mean(acc_c))]);
   end
end

instance = 1;
if instance

   % generate training and test indexes
   indextrain = 1:length(rgbdilabel);
   indextest = find(rgbdvlabel == 2);
   indextrain(indextest) = [];

   % generate training and test samples
   load rgbdfea_rgb_lbpkdes;
   trainhmp = rgbdfea(:, indextrain);
   trainlabel = rgbdilabel(:, indextrain);
   clear rgbdfea;
   [trainhmp, minvalue, maxvalue] = scaletrain(trainhmp, 'power');

   disp('Performing liblinear ... ...');
   lc = 10;
   % classify with liblinear
   option = ['-s 1 -c ' num2str(lc)];
   model = train(trainlabel',trainhmp',option);
   load rgbdfea_rgb_lbpkdes;
   testhmp = rgbdfea(:, indextest);
   testlabel = rgbdilabel(:, indextest);
   clear rgbdfea;
   testhmp = scaletest(testhmp, 'power', minvalue, maxvalue);
   [predictlabel, accuracy, decvalues] = predict(testlabel', testhmp', model);
   acc_i = mean(predictlabel == testlabel');
   save('./results/rgb_lbpkdes_acc_i.mat', 'acc_i', 'predictlabel', 'testlabel');

   % print and save classification accuracy
   disp(['Accuracy of Liblinear is ' num2str(mean(acc_i))]);
end

