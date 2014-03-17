
%typical inputs:
% train_component_dir = '~/dpd/dpd_scratch/cropped_bird_imgs_ffld/train/component';
% test_component_dir  = '~/dpd/dpd_scratch/cropped_bird_imgs_ffld/test/component';
% num_train = 5994;
% num_test  = 5794;
function [num_parts, train_component, test_component] = get_dpm_component_IDs(train_component_dir, test_component_dir, num_train, num_test)

    num_parts = 8; %default -- to return

    train_component = get_dpm_components_one_subset(train_component_dir, num_train);
    test_component = get_dpm_components_one_subset(test_component_dir, num_test);
end

%for train OR test
function components = get_dpm_components_one_subset(component_dir, num_imgs)
    components = zeros([1 num_imgs]);

    %TODO: separate function
    for i=1:num_imgs
        try
            %e.g. dpd_scratch/.../train/component/05994.txt
            in_fname = sprintf('%s/%05d.txt', component_dir, i);
            components(i) = csvread(in_fname); %(tweaked FFLD to output components indexed from 1 instead of 0)
        catch
            %no detection on this image
            components(i) = -1;
        end

    end

end

