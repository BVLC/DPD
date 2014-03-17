function mkdir_bo(datadir)
% make a directory with checking whether there is an existing directory
% written by Liefeng Bo in University of Washington on 01/04/2011

if exist(datadir,'dir')
   ;
else
   mkdir(datadir);
end

