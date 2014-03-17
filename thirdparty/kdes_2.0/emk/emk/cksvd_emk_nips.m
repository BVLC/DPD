function imfea = cksvd_emk_nips(feaSet, words, G, pyramid, ktype, kparam)
% extract image features using constrained kernel SVD
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

% set the parameters 
wordsnum = size(words,2);
pgrid = pyramid.^2;
sgrid = sum(pgrid);
weights = (1./pgrid); % divided by the number of grids at the coresponding level
weights = weights/sum(weights); 
imfea = zeros(sgrid*wordsnum,1);

patchsize = length(feaSet.feaArr);
for pp = 1:patchsize
    sift = double(feaSet.feaArr{pp});
    kz{pp} = eval_kernel(sift',words',ktype,kparam);
end

% spatial pyramid match with the learned low dimensional kernel
for s = 1:length(pyramid)
    wleng = feaSet.width/pyramid(s);
    hleng = feaSet.height/pyramid(s);
    xgrid = ceil(feaSet.x/wleng);
    ygrid = ceil(feaSet.y/hleng);
    allgrid = (ygrid -1 )*pyramid(s) + xgrid;
    pimimfea = zeros(pgrid(s)*wordsnum,1);
    for t = 1:pgrid(s)

        % find sift localized in the corresponding pyramid grid            
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
           pimimfea((t-1)*wordsnum+(1:wordsnum)) = G*(mkzind');
        else
           pimimfea((t-1)*wordsnum+(1:wordsnum)) = 0;
        end
    end
    imfea(sum(pgrid(1:s-1))*wordsnum + (1:pgrid(s)*wordsnum),1) = weights(s)*pimimfea;
end

