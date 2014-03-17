function words = visualwords(fea_params, basis_params) 
% generate basis vectors/visual words
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 

disp('Sample kernel descriptors  ... ...');
kdessample = sample_kdes(fea_params.feapath, basis_params.samplenum);

disp('Perform K-Means ... ...');
% kmeans
words = kmeans_bo(kdessample', basis_params.wordnum);
words = words';


