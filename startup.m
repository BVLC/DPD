%add local paths for DPD subdirectories
% adapted from voc-release5 DPM code startup.m
global G_STARTUP;
if isempty(G_STARTUP)
    G_STARTUP = true;

    % Avoiding addpath(genpath('.')) because .git includes
    % a VERY large number of subdirectories, which makes 
    % startup slow
  
    incl = {'thirdparty', 'data', 'dpm', 'common', 'dpd_kdes', ...
            'thirdparty/kdes_2.0/liblinear-1.5-dense-float' ... %add liblinear 1.5 to the path last, so that it's the default 
           };
    for i = 1:length(incl)
        addpath(genpath(incl{i}));
    end
end


