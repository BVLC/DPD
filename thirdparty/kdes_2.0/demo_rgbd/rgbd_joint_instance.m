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
load rgbdfea_rgb_nrgbkdes.mat;
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

instance = 1;
if instance

   % generate training and test indexes
   indextrain = 1:length(rgbdilabel);
   indextest = find(rgbdvlabel == 2);
   indextrain(indextest) = [];

   % generate training and test samples
   load rgbdfea_joint;
   trainfea = rgbdfea_joint(:, indextrain);
   trainlabel = rgbdilabel(:, indextrain);
   clear rgbdfea_joint;

   disp('Performing liblinear ... ...');
   [trainfea, minvalue, maxvalue] = scaletrain(trainfea, 'power');
   lc = 10;
   % classify with liblinear
   option = ['-s 1 -c ' num2str(lc)];
   model = train(trainlabel',trainfea',option);
   load rgbdfea_joint;
   testfea = rgbdfea_joint(:, indextest);
   testlabel = rgbdilabel(:, indextest);
   clear rgbdfea_joint;
   testfea = scaletest(testfea, 'power', minvalue, maxvalue);
   [predictlabel, accuracy, decvalues] = predict(testlabel', testfea', model);
   acc_i = mean(predictlabel == testlabel');
   save('./results/joint_acc_i.mat', 'acc_i', 'predictlabel', 'testlabel');

   % print and save classification accuracy
   disp(['Accuracy of Liblinear is ' num2str(mean(acc_i))]);
end


