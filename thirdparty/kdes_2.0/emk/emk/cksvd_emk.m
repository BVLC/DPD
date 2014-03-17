function imfea = cksvd_emk(fea_params, basis_params, emk_params)
% extract image features using constrained kernel SVD
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

% set the parameters 
basisnum = size(basis_params.basis,2);
pgrid = emk_params.pyramid.^2;
sgrid = sum(pgrid);
weights = (1./pgrid); % divided by the number of grids at the coresponding level
weights = weights/sum(weights); 
imfea = zeros(sgrid*basisnum,1);

patchsize = length(fea_params.feaSet.feaArr);
for pp = 1:patchsize
    kdes = double(fea_params.feaSet.feaArr{pp});
    kz{pp} = eval_kernel(kdes', basis_params.basis', emk_params.ktype, emk_params.kparam);
end

% spatial pyramid match with the learned low dimensional kernel
for s = 1:length(emk_params.pyramid)
    wleng = fea_params.feaSet.width/emk_params.pyramid(s);
    hleng = fea_params.feaSet.height/emk_params.pyramid(s);
    xgrid = ceil(fea_params.feaSet.x/wleng);
    ygrid = ceil(fea_params.feaSet.y/hleng);
    allgrid = (ygrid -1 )*emk_params.pyramid(s) + xgrid;
    pimimfea = zeros(pgrid(s)*basisnum,1);
    for t = 1:pgrid(s)

        % find kdes localized in the corresponding pyramid grid            
        ind = find(allgrid == t);
        if length(ind)
           kzind = [];
           for pp = 1:patchsize
               kzind = [kzind; kz{pp}(ind,:)];
           end

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

