clear;

% add paths
addpath('../liblinear-1.5-dense-float/matlab');
addpath('../helpfun');
addpath('../kdes');
addpath('../emk');

% combine all kernel descriptors
rgbdfea_joint = [];
load rgbdfea_rgb_gradkdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
load rgbdfea_rgb_lbpkdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
load rgbdfea_depth_gradkdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
load rgbdfea_depth_lbpkdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
load rgbdfea_pcloud_spinkdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
load rgbdfea_pcloud_sizekdes.mat;
rgbdfea_joint = [rgbdfea_joint; rgbdfea];
save -v7.3 rgbdfea_joint rgbdfea_joint rgbdclabel rgbdilabel rgbdvlabel;

category = 1;
if category
   trail = 5;
   for i = 1:trail
       % generate training and test samples
       ttrainindex = [];
       ttestindex = [];
       labelnum = unique(rgbdclabel);
       for j = 1:length(labelnum)
           trainindex = find(rgbdclabel == labelnum(j));
           rgbdilabel_unique = unique(rgbdilabel(trainindex));
           perm = randperm(length(rgbdilabel_unique));
           subindex = find(rgbdilabel(trainindex) == rgbdilabel_unique(perm(1)));
           testindex = trainindex(subindex);
           trainindex(subindex) = [];
           ttrainindex = [ttrainindex trainindex];
           ttestindex = [ttestindex testindex];
       end
       load rgbdfea_joint;
       trainfea = rgbdfea_joint(:,ttrainindex);
       clear rgbdfea_joint;
       [trainfea, minvalue, maxvalue] = scaletrain(trainfea, 'power'); 
       trainlabel = rgbdclabel(ttrainindex); % take category label

       % classify with liblinear
       lc = 10;
       option = ['-s 1 -c ' num2str(lc)];
       model = train(trainlabel',trainfea',option);
       load rgbdfea_joint;
       testfea = rgbdfea_joint(:,ttestindex);
       clear rgbdfea_joint;
       testfea = scaletest(testfea, 'power', minvalue, maxvalue);
       testlabel = rgbdclabel(ttestindex); % take category label
       [predictlabel, accuracy, decvalues] = predict(testlabel', testfea', model);
       acc_c(i,1) = mean(predictlabel == testlabel');
       save('./results/joint_acc_c.mat', 'acc_c', 'predictlabel', 'testlabel');

       % print and save results
       disp(['Accuracy of Liblinear is ' num2str(mean(acc_c))]);
   end
end

