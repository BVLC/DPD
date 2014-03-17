function AP = dpd_classify_human(config, is_strong, test_only)

if ~test_only
    load([config.save_feature_path '_train.mat']);
    [trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
    for i = 1:config.num_attributes
        fprintf('Train linear SVM for attribute %d\n',i);
        lc = 1; % regularization parameter C
        option = ['-s 1 -c ' num2str(lc)];
        train_index = find(config.trainlabel{i}~=0);
        model{i} = train(config.trainlabel{i}(train_index),single(trainkdes(:,train_index))',option);
    end
    clear trainkdes;
    if is_strong
        save(['data/model_' config.database '_strong' ],'model','minvalue','maxvalue');
    else
        save(['data/model_' config.database '_weak' ],'model','minvalue','maxvalue');
    end
else
    if is_strong
        load(['data/model_' config.database '_strong']);
    else
        load(['data/model_' config.database '_weak']);
    end
end

load([config.save_feature_path '_test.mat']);
testkdes = scaletest(testkdes, 'power', minvalue, maxvalue);
sum_AP = 0;
for i = 1:config.num_attributes
    test_index = find(config.testlabel{i}~=0);
    [~, ~, decvalues] = predict(config.testlabel{i}(test_index), single(testkdes(:,test_index))', model{i});
    decvalues = decvalues * model{i}.Label(1);
    sorted_score{i}= sortrows([decvalues config.testlabel{i}(test_index)]);
    AP{i} = plotBinaryPR(sorted_score{i});
    sum_AP = sum_AP + AP{i};
end
fprintf('The average AP is %f\n', sum_AP/config.num_attributes);
clear testkdes;
end
