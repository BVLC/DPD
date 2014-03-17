%crop out the pixels for each part, and save a small image for each part

[config, kdes_config] = dpd_set_up('bird',0);
[num_parts,train_component, test_component, train_parts, test_parts] ...
    = get_dpm_detections(config);

root = '/tscratch/tmp/nzhang/bird_parts/';
for p = 1:8
    mkdir([root 'train_part_' num2str(p)])
    for i =28:numel(config.impathtrain)
        part = round(train_parts{p}(i,:));
        if(part(1)==-1)
            continue
        end
        img = imread(config.impathtrain{i});
        i
        img_parts = img(max(1,part(2)):min(part(4),size(img,1)), max(1,part(1)):min(part(3),size(img,2)) ,:);
        img_parts = imresize(img_parts,[256 256]);
        filename = [root 'train_part_' num2str(p) '/' sprintf('%04d',i)];
        imwrite(img_parts,filename,'jpg');
        if mod(i,100)==1
            i
        end
    end
end
