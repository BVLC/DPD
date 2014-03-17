function [pclouds, distance] = depthtocloud(depth, topleft)
% convert depth maps to 3d point clouds
%
%-inputs
% depth         -depth map
% topleft       -topleft coordinates of the segmented image in the whole image
%
%-outputs
% pclouds       -3d point clouds

% written by Liefeng Bo on 2012

if nargin < 2
    topleft = [1 1];
end

depth = double(depth);
% Primesense constants
center = [320 240];
[imh, imw] = size(depth);
constant = 570.3;

% convert depth image to 3d point clouds
pclouds = zeros(imh,imw,3);
xgrid = ones(imh,1)*(1:imw) + (topleft(1)-1) - center(1);
ygrid = (1:imh)'*ones(1,imw) + (topleft(2)-1) - center(2);
pclouds(:,:,1) = xgrid.*depth/constant;
pclouds(:,:,2) = ygrid.*depth/constant;
pclouds(:,:,3) = depth;
distance = sqrt(sum(pclouds.^2,3));

