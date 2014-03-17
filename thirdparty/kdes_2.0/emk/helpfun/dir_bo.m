function imname = dir_bo(datadir)
% written by Liefeng Bo in University of Washington on 01/04/2011

% remove rootdir
imname = dir(datadir);
imname(1:2) = [];

