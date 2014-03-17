function savedir = gen_kdes_all(data_params, kdes_params)
% compute all kernel descriptors defined by kdes_params
% written by Liefeng Bo on March 27, 2012

% create save paths
for i = 1:length(kdes_params.kdestype)
    savedir{i} = [data_params.savedir kdes_params.kdestype{i}];
    mkdir_bo(savedir{i});
    kdes_save{i} = [kdes_params.kdestype{i} '_params.mat'];
end

% extract all kernel descriptors
for i = 1:length(kdes_params.kdestype)
    kdespath = get_kdes_path(savedir{i});
    if ~length(kdespath)
      kdes_struct = load(kdes_save{i});
      kdes_name = fieldnames(kdes_struct);
      kdes = getfield(kdes_struct, [kdes_params.kdestype{i} '_params']);
      kdes_params.kdes = kdes;
      data_params.savedir = savedir{i};
      gen_kdes_batch(data_params, kdes_params); 
    end
end

