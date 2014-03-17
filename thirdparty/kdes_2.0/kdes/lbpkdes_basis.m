function [bv, lbppoints, spoints] = lbpkdes_basis()
% generate basis vectors for local binary pattent basis vectors
% written by Liefeng Bo on Jan. 03, 2011

% uniformly and densely sample the spatial positions
sx = linspace(0, 1, 5);
sy = linspace(0, 1, 5);

% sample all local binary patterns
lbp1 = [0 1];
lbp2 = [0 1];
lbp3 = [0 1];
lbp4 = [0 1];
lbp5 = [0 1];
lbp6 = [0 1];
lbp7 = [0 1];
lbp8 = [0 1];

% generate the joint basis vectors using Kronecker tensor product
[bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8, bv9, bv10] = ndgrid(lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, sx, sy);
bv = [bv1(:)'; bv2(:)'; bv3(:)'; bv4(:)'; bv5(:)'; bv6(:)'; bv7(:)'; bv8(:)'; bv9(:)'; bv10(:)'];

% generate the local binary pattern basis vectors
[bv1, bv2, bv3, bv4, bv5, bv6, bv7, bv8] = ndgrid(lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8);
lbppoints = [bv1(:)'; bv2(:)'; bv3(:)'; bv4(:)'; bv5(:)'; bv6(:)'; bv7(:)'; bv8(:)'];

% generate the spatial basis vectors
[bv1, bv2] = ndgrid(sx, sy);
spoints = [bv1(:)'; bv2(:)'];

