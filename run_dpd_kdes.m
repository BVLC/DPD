function accuracy = run_dpd(database, is_strong, test_only)
%%%Written by Ning Zhang, Aug 28,2013
%%main script to run dpd
%%Params: database(string) bird is cub200-2011 
%%                 cub200 is cub200-2010
%%                 human is berkeley attribute 
%%        is_strong(boolean) true means strong-DPD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[config, kdes_config] = dpd_set_up(database, is_strong);

%%%POOLING%%%%
dpd_pooling(config, kdes_config, is_strong);

%%%CLASSIFICATION%%%
if strcmp(database, 'bird') || strcmp(database, 'cub200')
    accuracy = dpd_classify(config, is_strong, test_only);
else
    accuracy = dpd_classify_human(config, is_strong, test_only);
end
