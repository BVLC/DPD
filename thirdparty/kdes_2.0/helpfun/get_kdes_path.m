function kdespath = get_kdes_dir(kdesdir)
% generate the image paths
% written by Liefeng Bo on 01/05/2011 in University of Washington

% subdirectory
kdesname = dir_bo(kdesdir);
if length(kdesname)
   for i = 1:length(kdesname)
       % generate image paths
       kdespath{1,i} = [kdesdir '/' kdesname(i).name];
   end
else
   kdespath = [];
end

