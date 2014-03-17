function accuracy = dpd_classify(config, is_strong, test_only)
if ~test_only
    load([config.save_feature_path '_train.mat']);
    [trainkdes, minvalue, maxvalue] = scaletrain(trainkdes, 'power');
    disp('Train linear SVM ... ...');
    lc = 1; % regularization parameter C
    option = ['-s 1 -c ' num2str(lc)];
    model = train(config.trainlabel,single(trainkdes)',option);
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
[pred_label, accuracy, ~] = predict(config.testlabel, single(testkdes)', model);

if is_strong
    save(['data/' config.database '_pred_labels_strong.mat'], 'pred_label', 'accuracy');
else
    save(['data/' config.database '_pred_labels_weak.mat'], 'pred_label', 'accuracy');
end

end
