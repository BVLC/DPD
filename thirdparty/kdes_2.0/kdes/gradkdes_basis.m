function [bv, gpoints, spoints] = gradkdes_basis()
% generate basis vectors for gradient kernel descripotrs
% written by Liefeng Bo on Jan. 03, 2011

% uniformly and densely sample the spatial positions
sx = linspace(0, 1, 5);
sy = linspace(0, 1, 5);

% uniformly and densely sample the gradient orientations
gx = linspace(-1, 1, 10);
gy = linspace(-1, 1, 10);
% theta = linspace(0, 2*pi, 50);

% generate the joint basis vectors by Kcronecker tensor product
[bv1, bv2, bv3, bv4] = ndgrid(gx,gy,sx,sy);
bv = [bv1(:)'; bv2(:)'; bv3(:)'; bv4(:)'];
% [bv1, bv2, bv3] = ndgrid(theta,x,y);
% bv = [bv [sin(bv1(:)'); cos(bv1(:)'); bv2(:)'; bv3(:)']];

% generate the orientation basis vectors
[bv1, bv2] = ndgrid(gx, gy);
gpoints = [bv1(:)'; bv2(:)'];
% gpoints = [sin(theta(:)'); cos(theta(:)')];

% generate the spatial basis vectors
[bv1, bv2] = ndgrid(sx, sy);
spoints = [bv1(:)'; bv2(:)'];

