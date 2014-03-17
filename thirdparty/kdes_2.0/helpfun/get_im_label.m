function [imlabel, impath] = get_im_label(imdir, subname)
% generate the image paths and the corresponding image labels
% written by Liefeng Bo on 01/05/2011 in University of Washington

% directory
if nargin < 2
   subdir = dir_bo(imdir);
   it = 0;
   for i = 1:length(subdir)
       datasubdir{i} = [imdir '/' subdir(i).name];
       dataname = dir_bo(datasubdir{i});
       for j = 1:size(dataname,1)
           it = it + 1;
 	   % generate image paths
	   impath{1,it} = [datasubdir{i} '/' dataname(j).name];
           imlabel(1,it) = i;
       end
   end

else
   subdir = dir_bo(imdir);
   it = 0;
   for i = 1:length(subdir)
       datasubdir{i} = [imdir subdir(i).name];
       dataname = dir([datasubdir{i} '/*' subname]);
       for j = 1:size(dataname,1)
           it = it + 1;
           % generate image paths
           impath{1,it} = [datasubdir{i} '/' dataname(j).name];
           imlabel(1,it) = i;
       end
   end
end



