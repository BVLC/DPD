function [bv, rgbpoints, spoints] = rgbkdes_basis()
% generate basis vectors for color kernel descriptors
% written by Liefeng Bo on Jan. 03, 2011

% uniformly and densely sample the spatial positions
sx = linspace(0,1,5);
sy = linspace(0,1,5);

% uniformly and densely sample the rgb channels
r = linspace(0,1,5);
g = linspace(0,1,5);
b = linspace(0,1,5);

% generate the joint basis vectors using Kronecker basis vectors
[bv1, bv2, bv3, bv4, bv5] = ndgrid(r, g, b, sx, sy);
bv = [bv1(:)'; bv2(:)'; bv3(:)'; bv4(:)'; bv5(:)'];

% generate the color basis vectors
[bv1, bv2, bv3] = ndgrid(r,g,b);
rgbpoints = [bv1(:)'; bv2(:)'; bv3(:)'];

% generate the spatial basis vectors
[bv1, bv2] = ndgrid(sx,sy);
spoints = [bv1(:)'; bv2(:)'];


