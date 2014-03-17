function words = visualwords(feapath, samplenum, wordnum) 
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

disp('Sample local features  ... ...');
feasample = sample_fea(feapath, samplenum);
disp('Perform K-Means ... ...');

% matlab kmeans
words = kmeans_bo(feasample', wordnum);
words = words';

function feaset = sample_fea(feapath, samplenum)
% sample fea for clustering
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

load(feapath{1});
dim = size(feaSet.feaArr{1},1);
scalesize = length(feaSet.feaArr);

% initialize the parameters
samsize = length(feapath)*samplenum*scalesize;
feaset = zeros(dim,samsize);
index = [];
it = 0;
count = 0;
for i = 1:length(feapath)
    load(feapath{i});
    for ss = 1:scalesize
        it = it + 1;
        fea = double(feaSet.feaArr{ss});
        feanum = size(fea,2);
        num = min(feanum,samplenum);
        perm = randsample(feanum, num);
        feaset(:,count+(1:num)) = fea(:,perm);
        count = count + num;
        if mod(it,10) == 1
           disp(['Current Iteration is: ' num2str(it)]);
        end
    end
end
feaset(:,count+1:end) = [];

