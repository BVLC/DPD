function [num_part, train_component, test_component, train_parts, test_parts] ...
    = get_dpm_detections(config)

num_train = numel(config.impathtrain);
num_test = numel(config.impathtest);
num_part = 8;  %default value

load(fullfile([config.dpm_detection_path '_train']))
train_component = zeros(num_train,1);
for i = 1 : num_train
    if ~isempty(parts1{i})
        train_component(i) = parts1{i}(end-1);
        num_part = (numel(parts1{i})-2) / 4 -1;
        for t = 1:num_part
            train_parts{t}(i,:) = parts1{i}(1+4*t:4*(t+1));
        end
    else
        train_component(i) =-1;
        for t = 2:num_part+1
            train_parts{t-1}(i,:) = [-1 -1 -1 -1];
        end
    end
end
load(fullfile([config.dpm_detection_path '_test']))
test_component = zeros(num_test,1);
for i = 1:num_test
    if ~isempty(parts1{i})
        test_component(i) = parts1{i}(end-1);
        for t = 1:num_part
            test_parts{t}(i,:) = parts1{i}(1+4*t:4*(t+1));
        end
    else
        test_component(i) =-1;
        for t = 2:num_part+1
            test_parts{t-1}(i,:) = [-1 -1 -1 -1];
        end
    end
end
end
