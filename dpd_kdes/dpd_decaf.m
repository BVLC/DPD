    addpath(genpath('/u/vis/nzhang/projects/birdmix/code_release/matlab/'));

    [config,  ~] = dpd_set_up('bird',0);
    doTrain = 0;

    %TODO: remove/simplify get_dpm_detections() -- we only use (num_parts and component_IDs). the actual parts were already used to compute per-part DeCAF features    
    %[num_parts,train_component, test_component, train_parts, test_parts] = get_dpm_detections(config);

    % TODO: move these params to dpd_set_up
    train_component_dir = '~/dpd/dpd_scratch/cropped_bird_imgs_ffld/train/component';
    test_component_dir  = '~/dpd/dpd_scratch/cropped_bird_imgs_ffld/test/component';
    num_train = 5994;
    num_test  = 5794;
    [num_parts, train_component, test_component] = get_dpm_component_IDs(train_component_dir, test_component_dir, num_train, num_test);

  %TRAIN
    if(doTrain)
        qqq = load([config.scratch_dir '/decaf_bird_features_forrest/feature_bbox_train.mat']);
        fea = qqq.features;

        for p = 1: num_parts
            qqq = load([config.scratch_dir '/decaf_bird_features_forrest/feature_train_part_' num2str(p) '.mat']);
            fea = [fea qqq.features];
        end

        basis = size(fea,2) / (num_parts + 1);

        num_regions = size(config.weak_weights,1);
        fea_region = cell(num_regions,1);
        for r = 1 : num_regions
            fea_region{r} = zeros(size(fea,1), basis);
        end
        for c = 1 : config.num_components
            c_index = find(train_component == c);
            for p = 1 : num_parts
                for r = 1 : num_regions
                    fea_region{r}(c_index,:) = ...
                    fea_region{r}(c_index,:) + config.weak_weights(r,p,c)*fea(c_index,p*basis+1:(p+1)*basis);
                end
            end
        end

        fea_weak = fea(:,1:basis);
        for r = 1:num_regions
            fea_weak = [fea_weak  fea_region{r} ]; %TODO: semicolon and single() cast?
        end
        train_fea = single(fea_weak);
        [train_fea, minvalue, maxvalue] = scaletrain(train_fea, 'power');
        disp('Train linear SVM ... ...');
        lc = 1; % regularization parameter C
        option = ['-s 1 -c ' num2str(lc)];
        model = train(double(config.trainlabel), single(train_fea), option);
        save('data/model_dpd_decaf_bird_weak.mat','model', 'minvalue', 'maxvalue');
    else
        load('data/model_dpd_decaf_bird_weak.mat');
    end

    qqq = load([config.scratch_dir '/decaf_bird_features_forrest/feature_bbox_test.mat']);
    fea = qqq.features;

  %TEST

    for p = 1: num_parts
        %qqq = load(['feature_test_part_' num2str(p) '.mat']);
        qqq = load([config.scratch_dir '/decaf_bird_features_forrest/feature_test_part_' num2str(p) '.mat']);
        fea = [fea qqq.features];
    end

    basis = size(fea,2) / (num_parts + 1);
    num_regions = size(config.weak_weights,1);
    fea_region = cell(num_regions,1);
    for r = 1 : num_regions
        fea_region{r} = zeros(size(fea,1), basis);
    end
    for c = 1 : config.num_components
        c_index = find(test_component == c);
        for p = 1 : num_parts
            for r = 1 : num_regions
                fea_region{r}(c_index,:) = ...
                fea_region{r}(c_index,:) + config.weak_weights(r,p,c)*fea(c_index,p*basis+1:(p+1)*basis);
            end
        end
    end

    fea_weak = fea(:,1:basis);
    for r = 1:num_regions
        fea_weak = [fea_weak  fea_region{r} ];
    end
    test_fea = fea_weak;
    test_fea = test_fea(1:5794,:);
    test_fea = scaletest(test_fea, 'power', minvalue, maxvalue);

    disp('Test linear SVM ... ...');
    [~, accuracy, ~] = predict(double(config.testlabel), single(test_fea), model)
    save('accuracy_dpd_decaf','accuracy');


