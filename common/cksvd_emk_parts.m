function imfea = cksvd_emk_parts(fea_params, basis_params, emk_params, bbox,parts)
% extract image features using constrained kernel SVD
% written by Ning Zhang on 03/07/2013

%boundingbox_fea
pooling_box = bbox;
pooling_index = find(fea_params.feaSet.x>=pooling_box(1) &...
    fea_params.feaSet.x <= pooling_box(3) &...
    fea_params.feaSet.y >= pooling_box(2) &...
    fea_params.feaSet.y <= pooling_box(4));

kdes = double(fea_params.feaSet.feaArr{1});
fea_pooling.feaArr = fea_params.feaSet.feaArr{1}(:,pooling_index);
fea_pooling.x = fea_params.feaSet.x(pooling_index)-pooling_box(1);
fea_pooling.y = fea_params.feaSet.y(pooling_index)-pooling_box(2);
fea_pooling.height = pooling_box(4) -pooling_box(2);
fea_pooling.width = pooling_box(3) - pooling_box(1);
kz = eval_kernel(kdes', basis_params.basis', emk_params.ktype, emk_params.kparam);
kz_pooling = kz(pooling_index,:);


%divide kz into kz_head, kz_body, kz_bb then output to the pyramid
%change the spatial pyramid pooling to pooling on head/body/background

imfea = spatial_pyramid(kz_pooling,fea_pooling,emk_params,basis_params);
%kz = eval_kernel(kdes', basis_params.basis', emk_params.ktype, emk_params.kparam);

for p = 1:numel(parts)
    pooling_box = parts{p};
    pooling_index = find(fea_params.feaSet.x>=pooling_box(1) &...
        fea_params.feaSet.x <= pooling_box(3) &...
        fea_params.feaSet.y >= pooling_box(2) &...
        fea_params.feaSet.y <= pooling_box(4));
    fea_pooling.feaArr = fea_params.feaSet.feaArr{1}(:,pooling_index);
    fea_pooling.x = fea_params.feaSet.x(pooling_index)-pooling_box(1);
    fea_pooling.y = fea_params.feaSet.y(pooling_index)-pooling_box(2);
    fea_pooling.height = pooling_box(4) -pooling_box(2);
    fea_pooling.width = pooling_box(3) - pooling_box(1);  
    kz_pooling = kz(pooling_index,:);
    %divide kz into kz_head, kz_body, kz_bb then output to the pyramid
    %change the spatial pyramid pooling to pooling on head/body/background
    imfea_pooling = spatial_pyramid(kz_pooling,fea_pooling,emk_params,basis_params);
    imfea = [imfea ; imfea_pooling];
end
end


% spatial pyramid match with the learned low dimensional kernel
function imfea = spatial_pyramid(kz,feaSet,emk_params,basis_params)
basisnum = size(basis_params.basis,2);
pgrid = emk_params.pyramid.^2;
sgrid = sum(pgrid);
weights = (1./pgrid); % divided by the number of grids at the coresponding level
weights = weights/sum(weights);
patchsize = length(feaSet.feaArr);
for s = 1:length(emk_params.pyramid)
    wleng = feaSet.width/emk_params.pyramid(s);
    hleng = feaSet.height/emk_params.pyramid(s);
    xgrid = ceil(feaSet.x/wleng);
    ygrid = ceil(feaSet.y/hleng);
    allgrid = (ygrid -1 )*emk_params.pyramid(s) + xgrid;
    pimimfea = zeros(pgrid(s)*basisnum,1);
    for t = 1:pgrid(s)
        % find kdes localized in the corresponding pyramid grid
        ind = find(allgrid == t);
        if length(ind)
            kzind = [];
            kzind = [kzind; kz(ind,:)];          
            % suppress similar kernel descriptors using max
            [valueaaa, indaaa] = max(kzind,[],2);
            [valuebbb, indbbb] = sort(valueaaa,'descend');
            [valueccc, indccc] = unique(indaaa(indbbb),'first');
            indgrid = indbbb(indccc);
            mkzind = mean(kzind(indgrid,:),1);
            pimimfea((t-1)*basisnum+(1:basisnum)) = basis_params.G*(mkzind');
        else
            pimimfea((t-1)*basisnum+(1:basisnum)) = 0;
        end
    end
    imfea(sum(pgrid(1:s-1))*basisnum + (1:pgrid(s)*basisnum),1) = weights(s)*pimimfea;
end
end
