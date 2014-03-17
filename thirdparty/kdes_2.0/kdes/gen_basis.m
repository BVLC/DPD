function gen_basis()
% generate projection matrices for kernel descriptors
% written by Liefeng Bo on 2012

% gradient kernel descriptors
gradkdes_params.type = 'gradkdes';
gradkdes_params.method = 'kpca';
gradkdes_params.dim = 200;
gradkdes_params.ktype = 'rbf';
gradkdes_params.kparam = [5 5 3 3];
[gradkdes_params.points, gradkdes_params.gpoints, gradkdes_params.spoints] = gradkdes_basis();
[gradkdes_params.eigvectors, gradkdes_params.eigvalues, gradkdes_params.centervector] = kpca(gradkdes_params.points', gradkdes_params.dim, gradkdes_params.ktype, gradkdes_params.kparam);
gradkdes_params.ssize = size(gradkdes_params.spoints,2);
gradkdes_params.gsize = size(gradkdes_params.gpoints,2);
gradkdes_params.eigvectors = gradkdes_params.eigvectors';
save gradkdes_params gradkdes_params;
gradkdes_dep_params = gradkdes_params;
gradkdes_dep_params.type = 'gradkdes_dep';
save gradkdes_dep_params gradkdes_dep_params;

% local binary pattern kernel descriptors
lbpkdes_params.type = 'lbpkdes';
lbpkdes_params.method = 'kpca';
lbpkdes_params.dim = 200;
lbpkdes_params.ktype = 'rbf';
lbpkdes_params.kparam = [3*ones(1,8) 3 3];
[lbpkdes_params.points, lbpkdes_params.lbppoints, lbpkdes_params.spoints] = lbpkdes_basis();
lbpkdes_params.ssize = size(lbpkdes_params.spoints,2);
lbpkdes_params.lbpsize = size(lbpkdes_params.lbppoints,2);
[lbpkdes_params.eigvectors, lbpkdes_params.eigvalues, lbpkdes_params.centervector] = kpca(lbpkdes_params.points', lbpkdes_params.dim, lbpkdes_params.ktype, lbpkdes_params.kparam);
lbpkdes_params.eigvectors = lbpkdes_params.eigvectors';
save lbpkdes_params lbpkdes_params;
lbpkdes_dep_params = lbpkdes_params;
lbpkdes_dep_params.type = 'lbpkdes_dep';
save lbpkdes_dep_params lbpkdes_dep_params;

% rgb kernel descriptor
rgbkdes_params.type = 'rgbkdes';
rgbkdes_params.method = 'kpca';
rgbkdes_params.dim = 200;
rgbkdes_params.ktype = 'rbf';
rgbkdes_params.kparam = [5 5 5 3 3];
[rgbkdes_params.points, rgbkdes_params.rgbpoints, rgbkdes_params.spoints] = rgbkdes_basis();
rgbkdes_params.ssize = size(rgbkdes_params.spoints,2);
rgbkdes_params.rgbsize = size(rgbkdes_params.rgbpoints,2);
[rgbkdes_params.eigvectors, rgbkdes_params.eigvalues, rgbkdes_params.centervector] = kpca(rgbkdes_params.points', rgbkdes_params.dim, rgbkdes_params.ktype, rgbkdes_params.kparam);
rgbkdes_params.eigvectors = rgbkdes_params.eigvectors';
save rgbkdes_params rgbkdes_params;

% normalized rgb kernel descriptor
nrgbkdes_params.type = 'nrgbkdes';
nrgbkdes_params.method = 'kpca';
nrgbkdes_params.dim = 200;
nrgbkdes_params.ktype = 'rbf';
nrgbkdes_params.kparam = [5 5 5 3 3];
[nrgbkdes_params.points, nrgbkdes_params.rgbpoints, nrgbkdes_params.spoints] = rgbkdes_basis();
nrgbkdes_params.ssize = size(nrgbkdes_params.spoints,2);
nrgbkdes_params.rgbsize = size(nrgbkdes_params.rgbpoints,2);
[nrgbkdes_params.eigvectors, nrgbkdes_params.eigvalues, nrgbkdes_params.centervector] = kpca(nrgbkdes_params.points', nrgbkdes_params.dim, nrgbkdes_params.ktype, nrgbkdes_params.kparam);
nrgbkdes_params.eigvectors = nrgbkdes_params.eigvectors';
save nrgbkdes_params nrgbkdes_params;

% size kernel descriptors
sizekdes_params.type = 'sizekdes';
sizekdes_params.method = 'kpca';
sizekdes_params.dim = 50;
sizekdes_params.ktype = 'rbf';
sizekdes_params.kparam = 30;
sizekdes_params.points = sizekdes_basis();
[sizekdes_params.eigvectors, sizekdes_params.eigvalues, sizekdes_params.centervector] = kpca(sizekdes_params.points', sizekdes_params.dim, sizekdes_params.ktype, sizekdes_params.kparam);
sizekdes_params.eigvectors = sizekdes_params.eigvectors';
save sizekdes_params sizekdes_params;

% spin kernel descriptors
spinkdes_params.type = 'spinkdes';
spinkdes_params.method = 'kpca';
spinkdes_params.dim = 200;
spinkdes_params.ktype = 'rbf';
spinkdes_params.kparam = [3 3 300 300];
[spinkdes_params.points, spinkdes_params.npoints, spinkdes_params.spoints] = spinkdes_basis();
[spinkdes_params.eigvectors, spinkdes_params.eigvalues, spinkdes_params.centervector] = kpca(spinkdes_params.points', spinkdes_params.dim, spinkdes_params.ktype, spinkdes_params.kparam);
spinkdes_params.ssize = size(spinkdes_params.spoints,2);
spinkdes_params.nsize = size(spinkdes_params.npoints,2);
spinkdes_params.eigvectors = spinkdes_params.eigvectors';
save spinkdes_params spinkdes_params;




