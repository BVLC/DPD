function [config, kdes_params] = dpd_set_up(database, is_strong)
%%%%Written by Ning Zhang, Aug 28th, 2013
%%% Read images, labels and other set up
%%% Need changes before running on specific machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    startup %run dpd/startup.m

    scratch_dir = '/media/big_disk/dpd_scratch/'; %you edit this
    config.scratch_dir = scratch_dir;

    %%read data get impathtrain, impathtest, trainlabel, testlabel,train_bb, test_bb
    config.database = database;
    if strcmp(database, 'bird')
        %all this stuff comes with the off-the-shelf CUB_200_2011 dataset. (right?)

        dataset_base = '/media/big_disk/datasets/icsi_datasets/CUB_200_2011'; %you edit this
        img_base = [dataset_base '/images/'];
        imdir = [dataset_base '/images.txt'];
        [img_id img_path] = textread(imdir,'%d %s');
        traintest_dir = [dataset_base '/train_test_split.txt'];
        [img_id train_flag] = textread(traintest_dir, '%d %d');
        label_dir = [dataset_base '/image_class_labels.txt'];
        [img_id img_labels]=textread(label_dir,'%d %d');
        part_dir = [dataset_base '/parts/part_locs.txt'];  
        [img_id part part_x part_y visible] = textread(part_dir, '%d %d %f %f %d');
        boundingbox_dir = [dataset_base '/bounding_boxes.txt'];
        [img_id left top width height] = textread(boundingbox_dir, '%d %f %f %f %f');
        trainindex = find(train_flag == 1);
        testindex = find(train_flag == 0);
        config.trainlabel = img_labels(trainindex);
        config.testlabel = img_labels(testindex);
        config.impathtrain = strcat(img_base,img_path(trainindex));
        config.impathtest = strcat(img_base, img_path(testindex));
        for j = 1:length(config.impathtrain)
            i = trainindex(j);
            config.train_bb(j,:) = [left(i) top(i) left(i)+width(i) top(i)+height(i)];
        end
        for j = 1:length(config.impathtest)
            i = testindex(j);
            config.test_bb(j,:) = [left(i) top(i) left(i)+width(i) top(i)+height(i)];
        end
        
    elseif strcmp(database, 'cub200')
        dataset_base = '/u/vis/x1/common/CUB_200_2010'; %you edit this
        img_base = [dataset_base '/images/'];
        %class_name = textread('/u/vis/x1/common/CUB_200_2010/lists/classes.txt','%s');
        class_name = textread([dataset_base '/lists/classes.txt'],'%s');
        train_dir = [dataset_base '/lists/train.txt'];
        impathtrain_base = textread(train_dir,'%s');
        for i = 1:length(impathtrain_base)
            config.trainlabel(i) = str2num(impathtrain_base{i}(1:3));
        end
        config.impathtrain = strcat(img_base,impathtrain_base);
        
        test_dir = [dataset_base '/lists/test.txt'];
        impathtest_base = textread(test_dir,'%s');
        for i = 1:length(impathtest_base)
            config.testlabel(i) = str2num(impathtest_base{i}(1:3));
        end
        config.impathtest = strcat(img_base,textread(test_dir, '%s'));
        
        boundingbox_dir = [dataset_base '/annotations-mat/'];
        
        for i = 1:length(config.impathtrain)
            filename = [boundingbox_dir impathtrain_base{i}];
            filename(end-2:end)='mat';
            load(filename)
            config.train_bb(i,:)=[bbox.left bbox.top bbox.right bbox.bottom];
        end
        
        for i = 1:length(config.impathtest)
            filename = [boundingbox_dir impathtest_base{i}];
            filename(end-2:end)='mat';
            load(filename)
            config.test_bb(i,:)=[bbox.left bbox.top bbox.right bbox.bottom];
        end
            
    elseif strcmp(database, 'human')
        img_base = '/u/vis/x1/common/attributes_dataset/';
        train_dir = '/u/vis/x1/common/attributes_dataset/train/labels.txt';
        [impathtrain_base train_x1 train_y1 train_width train_height trainlabel{1}...
            trainlabel{2} trainlabel{3} trainlabel{4} trainlabel{5} trainlabel{6}...
            trainlabel{7} trainlabel{8} trainlabel{9}]  = ...
            textread(train_dir,'%s %f %f %f %f %d %d %d %d %d %d %d %d %d');
        config.impathtrain = strcat(img_base,'train/', impathtrain_base);
        config.trainlabel = trainlabel;
        config.num_attributes = numel(trainlabel);
        test_dir = '/u/vis/x1/common/attributes_dataset/test/labels.txt';
        [impathtest_base test_x1 test_y1 test_width test_height testlabel{1}...
            testlabel{2} testlabel{3} testlabel{4} testlabel{5} testlabel{6}...
            testlabel{7} testlabel{8} testlabel{9}]  =...
            textread(test_dir,'%s %f %f %f %f %d %d %d %d %d %d %d %d %d');
        config.impathtest = strcat(img_base,'test/', impathtest_base);
        config.testlabel = testlabel;
        %TODO config.train_bb, config.test_bb
        for i = 1 : length(config.impathtrain)
            config.train_bb(i,:) = [train_x1(i) train_y1(i) train_x1(i)+train_width(i) train_y1(i)+train_height(i)];
        end
        for i = 1 : length(config.impathtest)
            config.test_bb(i,:) = [test_x1(i) test_y1(i) test_x1(i)+test_width(i) test_y1(i)+test_height(i)];
        end
    else
        fprintf('Database %s not supported!\n',database);
    end

    %% setting dpm detection path and feature save path
    if is_strong
        config.dpm_detection_path = ['dpm/' database '_strong'];
        config.save_feature_path = ['/tscratch/tmp/nzhang/' database '_strong' ];
        %config.save_feature_path = [scratch_dir 'dpd_models/' database '_strong' ]; %TODO: rename 'dpd_models' to 'dpd_features' 
    else
        config.dpm_detection_path = ['dpm/' database '_weak'];
        config.save_feature_path = ['/tscratch/tmp/nzhang/' database '_weak' ];
        %config.save_feature_path = [scratch_dir 'dpd_models/' database '_weak' ];
    end
    config.num_components = 6;
    %codebook
    load(['codebook_' database]);
    kdes_params.codebook = codebook;

    %weak weights
    if ~is_strong
        load(['weak_weights_' database]);
        config.weak_weights = RESULTS{3};
    end

    %%kdes package setting
    kdes_params.grid = 8;   % kdes is extracted every 8 pixels
    kdes_params.patchsize = 16;  % patch size
    kdes_params.kdestype{1} = 'gradkdes';
    kdes_params.kdestype{2} = 'lbpkdes';
    kdes_params.kdestype{3} = 'rgbkdes';
    kdes_params.kdestype{4} = 'nrgbkdes';
    % extract kernel descriptors on training set and test set
    data_params.tag = 1;
    data_params.minsize = 2;  % minimum size of image
    data_params.maxsize = 2000; % maximum size of image

    config.savedirtrain = [scratch_dir database 'train']; 
    config.savedirtest = [scratch_dir database 'test'];


    % initialize the parameters of emk
    kdes_params.emk_params.pyramid = [1 2 3];
    kdes_params.emk_params.ktype = 'rbf';
    kdes_params.kparam = [0.001, 0.01, 0.01 0.01];

    %initialize the parameters of basis vectors
    kdes_params.basis_params.samplenum = 10; % maximum sample number per image scale
    kdes_params.basis_params.wordnum = 1000; % number of visual words
    kdes_params.basis_params.cholesky = kdes_cholesky(kdes_params); %insert cholesky'd basis functions into kdes_params   
 
    kdes_params.data_params = data_params;
end


