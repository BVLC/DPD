function [bv, npoints, spoints] = normalkdes_basis()
% generate basis vector for normal kernel descriptors
% written by Liefeng Bo on Jan. 03, 2011

% generate the gradient basis vectors

% uniformly and densely sample the spatial positions
sx = linspace(-0.05, 0.05, 10);
sy = linspace(0, 0.05, 5);

% uniformly and densely sample the gradient orientations
% gx = linspace(-1, 1, 10);
% gy = linspace(-1, 1, 10);
theta = linspace(0, 2*pi, 50);

% generate the joint basis vectors by Kcronecker tensor product
% [bv1, bv2, bv3, bv4] = ndgrid(gx,gy,sx,sy);
% bv = [bv1(:)'; bv2(:)'; bv3(:)'; bv4(:)'];
[bv1, bv2, bv3] = ndgrid(theta, sx, sy);
bv = [sin(bv1(:)'); cos(bv1(:)'); bv2(:)'; bv3(:)'];

% generate the orientation basis vectors
% [bv1, bv2] = ndgrid(gx, gy);
% npoints = [bv1(:)'; bv2(:)'];
npoints = [sin(theta(:)'); cos(theta(:)')];

% generate the spatial basis vectors
[bv1, bv2] = ndgrid(sx, sy);
spoints = [bv1(:)'; bv2(:)'];

