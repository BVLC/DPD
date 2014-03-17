function plabel = lsvm_predict(testfea, model)
% test multi-class SVMs with one-vs-all coding
% written by Liefeng Bo on 01/04/2011 in University of Washington

pred = testfea*model.w + ones(size(testfea,1),1)*model.b;
[aaa, plabel] =  max(pred,[],2);
plabel = model.ulabel(plabel);

