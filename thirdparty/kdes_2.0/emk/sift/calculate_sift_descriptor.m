function [database, lenStat] = calculate_sift_descriptor(rt_img_dir, rt_data_dir, gridSpacing, patchSize, minImSize, maxImSize, imtype, nrml_threshold)
%==========================================================================
% usage: calculate the sift descriptors given the image directory
%
% inputs
% rt_img_dir        -image database root path
% rt_data_dir       -feature database root path
% gridSpacing       -spacing for sampling dense descriptors
% patchSize         -patch size for extracting sift feature
% minImSize         -minimum size of the input image
% maxImSize         -maximum size of the input image
% imtype            -type of the input image
% nrml_threshold    -low contrast normalization threshold
%
% outputs
% database      -directory for the calculated sift features
%
% Lazebnik's SIFT code is used.
% modified by Liefeng Bo on Apr. 15, 2010
%==========================================================================

if nargin < 5
   minImSize = 0;
end
if nargin < 6
   maxImSize = 1000;
end
if nargin < 7
   imtype = 'png';
end
if nargin < 8
   nrml_threshold = 0.8;
end

disp('Extracting SIFT features...');
subfolders = dir(rt_img_dir);

maxpatchSize = max(patchSize);

siftLens = [];

database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

it = 0;
for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_img_dir, subname, ['*.' imtype]));
        
        c_num = length(frames);           
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        siftpath = fullfile(rt_data_dir, subname);        
        if ~isdir(siftpath),
            % mkdir(siftpath);
        end;
        
        for jj = 1:c_num,
            imgpath = fullfile(rt_img_dir, subname, frames(jj).name);
            
            if strcmp(imgpath(end-2:end), 'txt'); I=load(imgpath); else; I = imread(imgpath); end

            if ndims(I) == 3,
                I = im2double(rgb2gray(I));
            else
                I = im2double(I);
            end;
            
            [im_h, im_w] = size(I);
            
            if max(im_h, im_w) > maxImSize,
                I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
                [im_h, im_w] = size(I);
            end;
            if min(im_h, im_w) < minImSize,
               I = imresize(I, minImSize/min(im_h, im_w), 'bicubic');
            end;
            
            % make grid sampling SIFT descriptors
            remX = mod(im_w-maxpatchSize,gridSpacing);
            offsetX = floor(remX/2)+1;
            remY = mod(im_h-maxpatchSize,gridSpacing);
            offsetY = floor(remY/2)+1;
    
            [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-maxpatchSize+1, offsetY:gridSpacing:im_h-maxpatchSize+1);

            tic;
            % find SIFT descriptors
            fea_arr = sp_find_sift_grid(I, gridX, gridY, patchSize, nrml_threshold);
            feaSet.feaArr= fea_arr;
            time = toc;
            fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches, time %f\n', ...
                     frames(jj).name, im_w, im_h, size(gridX, 2), size(gridX, 1), length(patchSize)*numel(gridX),time);

            feaSet.x = gridX(:) + maxpatchSize/2 - 0.5;
            feaSet.y = gridY(:) + maxpatchSize/2 - 0.5;
            feaSet.width = im_w;
            feaSet.height = im_h;

            it = it + 1;
            save([rt_data_dir '/' sprintf('%06d',it)], 'feaSet');
        end;    
    end;
end; 
lenStat = hist(siftLens, 100);

