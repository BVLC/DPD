[y,xt] = libsvmread('../heart_scale');
xt = full(xt);
model=train(y, xt)
[l,a]=predict(y, xt, model);

