function dpd_pooling(config, kdes_params, is_strong)

%%GET PARTS BOXES%%
[num_parts,train_component, test_component, train_parts, test_parts] ...
    = get_dpm_detections(config);

if  ~exist([config.save_feature_path '_train.mat'],'file')

    %%GET KDES DESCRIPTORS%%
    kdes_params.data_params.datapath = config.impathtrain;
    kdes_params.data_params.savedir = config.savedirtrain;
    kdes_params.savedirtrain = gen_kdes_batch_all(kdes_params.data_params, kdes_params);

    %%POOLING%%
    trainkdes = [];
    for i = 1:4
        fea_params.feapath = get_kdes_path(kdes_params.savedirtrain{i});
        kdes_params.basis_params.basis = kdes_params.codebook{i};
        kdes_params.basis_params.G = kdes_params.basis_params.cholesky(kdes_params.kdestype{i}); %get cholesky'd basis functions for this type of KDES feature (grad, lbp, ...)
        kdes_params.emk_params.kparam = kdes_params.kparam(i);
        pooling_params.bb = config.train_bb;
        pooling_params.parts = train_parts;
        fea = cksvd_emk_batch_dpm(fea_params, kdes_params.basis_params, kdes_params.emk_params,pooling_params); %fea(imgIdx, :)
        if ~is_strong
            basis = size(fea,1) / (num_parts + 1);
            num_regions = size(config.weak_weights,1);
            fea_region = cell(num_regions,1);
            for r = 1 : num_regions
                fea_region{r} = zeros(basis,size(fea,2));
            end
            for c = 1 : config.num_components
                c_index = find(train_component == c);
                for p = 1 : num_parts
                    for r = 1 : num_regions
                        %vectorized over many images
                        fea_region{r}(:,c_index) = fea_region{r}(:,c_index) + config.weak_weights(r,p,c)*fea(p*basis+1:(p+1)*basis,c_index);
                    end
                end
            end
            fea_weak = fea(1:basis,:);
            for r = 1:num_regions
                fea_weak = [fea_weak; fea_region{r}];
            end
            trainkdes = [trainkdes; single(fea_weak)];
            clear fea_region;
        else
            trainkdes = [trainkdes; single(fea)];
        end
    end
    save([config.save_feature_path '_train.mat'], 'trainkdes', '-v7.3');
end

if  ~exist([config.save_feature_path '_test.mat'],'file')

    %%GET KDES DESCRIPTORS%%
    kdes_params.data_params.datapath = config.impathtest;
    kdes_params.data_params.savedir = config.savedirtest;
    kdes_params.savedirtest = gen_kdes_batch_all(kdes_params.data_params, kdes_params);

    %%POOLING%%
    testkdes = [];
    for i = 1:4
        fea_params.feapath = get_kdes_path(kdes_params.savedirtest{i});
        kdes_params.basis_params.basis = kdes_params.codebook{i};
        kdes_params.basis_params.G = kdes_params.basis_params.cholesky(kdes_params.kdestype{i}); %get cholesky'd basis functions for this type of KDES feature (grad, lbp, ...)
        kdes_params.emk_params.kparam = kdes_params.kparam(i);
        pooling_params.bb = config.test_bb;
        pooling_params.parts = test_parts;
        fea = cksvd_emk_batch_dpm(fea_params, kdes_params.basis_params, kdes_params.emk_params,pooling_params);
        if ~is_strong
            basis = size(fea,1) / (num_parts + 1);
            num_regions = size(config.weak_weights,1);
            fea_region = cell(num_regions,1);
            for r = 1 : num_regions
                fea_region{r} = zeros(basis,size(fea,2));
            end
            for c = 1 : config.num_components
                c_index = find(test_component == c);
                for p = 1 : num_parts
                    for r = 1 : num_regions
                        fea_region{r}(:,c_index) = fea_region{r}(:,c_index) + config.weak_weights(r,p,c)*fea(p*basis+1:(p+1)*basis,c_index);
                    end
                end
            end
            fea_weak = fea(1:basis,:);
            for r = 1:num_regions
                fea_weak = [fea_weak; fea_region{r}];
            end
            testkdes = [testkdes; single(fea_weak)];
            clear fea_region;
        else
            testkdes = [testkdes; single(fea)];
        end
    end
    save([config.save_feature_path '_test.mat'], 'testkdes', '-v7.3');
end

end


%THE CRUX -- pool KDES features based on DPM part descriptor locations
%@param fea = KDES features
% other params carry over from dpd_pooling() input 
%TODO: bolt this into trainkdes and testkdes calculation
function pooled_kdes = do_pooling(config, kdes_params, is_strong, fea, pooling_params, num_parts, component)
    if ~is_strong
        basis = size(fea,1) / (num_parts + 1);
        num_regions = size(config.weak_weights,1);
        fea_region = cell(num_regions,1);
        for r = 1 : num_regions
            fea_region{r} = zeros(basis,size(fea,2));
        end
        for c = 1 : config.num_components
            c_index = find(component == c);
            for p = 1 : num_parts
                for r = 1 : num_regions
                    fea_region{r}(:,c_index) = fea_region{r}(:,c_index) + config.weak_weights(r,p,c)*fea(p*basis+1:(p+1)*basis,c_index);
                end
            end
        end
        fea_weak = fea(1:basis,:);
        for r = 1:num_regions
            fea_weak = [fea_weak  ;fea_region{r} ];
        end
        pooled_kdes = fea_weak;
    else
        pooled_kdes = single(fea);
    end
end

