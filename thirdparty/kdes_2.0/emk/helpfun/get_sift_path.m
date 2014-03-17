function siftpath = get_sift_dir(siftdir)
% generate the image paths and the corresponding image labels
% written by Liefeng Bo on 01/04/2011 in University of Washington

% subdirectory
siftname = dir_bo(siftdir);
if length(siftname)
   for i = 1:length(siftname)
       % generate image paths
       siftpath{1,i} = [siftdir '/' siftname(i).name];
   end
else
   siftpath = [];
end

