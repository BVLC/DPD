function gen_kdes_batch(data_params, kdes_params)
%==========================================================================
% compute kernel descriptors for given image path set
%
%-inputs
% data_params	-parameters of data
% kdes_params	-parameters of kernel descriptors
% written by Liefeng Bo on March 26, 2012

if exist(data_params.savedir,'dir')
   ;
else
   mkdir(data_params.savedir);
end

disp('Extracting Kernel Descriptors ...');
% extract dense kernel descriptors for each image
for i = 1:length(data_params.datapath)

    % compute kernel descriptors
    tic;
    switch kdes_params.kdes.type
         case {'gradkdes', 'lbpkdes', 'rgbkdes', 'nrgbkdes'}
              % read an image
              if strcmp(data_params.datapath{i}(end-2:end), 'txt'); I = load(data_params.datapath{i}); else; I = imread(data_params.datapath{i}); end

              % resize an image
              if data_params.tag
                 im_h = size(I,1);
                 im_w = size(I,2);
                 if max(im_h, im_w) > data_params.maxsize,
                    I = imresize(I, data_params.maxsize/max(im_h, im_w), 'bicubic');
                 end
                 if min(im_h, im_w) < data_params.minsize,
                    I = imresize(I, data_params.minsize/min(im_h, im_w), 'bicubic');
                 end
              end
              
              % extract dense kernel descriptors over images
              switch kdes_params.kdes.type
                   case 'gradkdes'
                         feaSet = gradkdes_dense(I, kdes_params);
                   case 'lbpkdes'
                        feaSet = lbpkdes_dense(I, kdes_params);
                   case 'rgbkdes'    
                        feaSet = rgbkdes_dense(I, kdes_params);
                   case 'nrgbkdes'            
                        feaSet = nrgbkdes_dense(I, kdes_params);
                   otherwise
                        disp('Unknown kernel descriptors');
              end

         case {'gradkdes_dep', 'lbpkdes_dep'}
              % read a depth map
              I = imread(data_params.datapath{i});
              % normalize depth values to meter
              I = double(I)/1000;
              % extract dense kernel descriptors over depth maps
              switch kdes_params.kdes.type
                   case 'gradkdes_dep'
                         feaSet = gradkdes_dense(I, kdes_params);
                   case 'lbpkdes_dep'
                        feaSet = lbpkdes_dense(I, kdes_params);
                   otherwise
                        disp('Unknown kernel descriptors');
              end

         case {'normalkdes', 'sizekdes'}
            % read a depth map
            I = imread(data_params.datapath{i});
            I = double(I);
            topleft = fliplr(load([data_params.datapath{i}(1:end-13) 'loc.txt']));
            pcloud = depthtocloud(I, topleft);
            % normalize depth values to meter
            pcloud = pcloud./1000;
            % extract dense kernel descriptors over point clouds
            switch kdes_params.kdes.type
                 case 'normalkdes'
                      feaSet = normalkdes_dense(pcloud, kdes_params);
                 case 'sizekdes'
                      feaSet = sizekdes_dense(pcloud, kdes_params);
                 otherwise
                      disp('Unknown kernel descriptors');
              end

         otherwise
            disp('Unknown kernel descriptors');
    end
    % compute feature extraction time
    time = toc;

    % save kernel descriptors
    save([data_params.savedir '/' sprintf('%06d',i)], 'feaSet');
     
    % print feature extraction information
    ind = find(data_params.datapath{i} == '/');
    fprintf('Image ID %s: width= %d, height= %d,  %d patches, time %f\n', data_params.datapath{i}(ind(end)+1:end), feaSet.width, feaSet.height, length(feaSet.x(:)),time);
end;


