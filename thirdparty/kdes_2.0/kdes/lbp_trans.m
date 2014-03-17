function lbp = lbp_trans(im)
% transform pixels to local binary pattern
% written by Liefeng Bo on 2010

% gray image
try
   im = rgb2gray(im);
catch
   im = im;
end
im = double(im);

% 3x3 windows
windows = [3 3];
cdim = prod(windows) - 1;
for i = 1:cdim
    ffilter{i} = zeros(windows);
end

% construct filters
it = 0;
for i = 1:windows(1)
    for j = 1:windows(2) 
        if ((i-1)*windows(2)+j) ~= round(prod(windows)/2)
           it = it + 1; 
           ffilter{it}(i,j) = 1;
        else
           ;
        end
    end
end

% generate local binary pattern
lbp = zeros(size(im,1),size(im,2),cdim+1);
for i = 1:cdim
    I_f(:,:,i) = filter2(ffilter{i}, im, 'same');
    lbp(:,:,i) = double(I_f(:,:,i) >= im); 
end
I_f(:,:,cdim+1) = im;
lbp(:,:,cdim+1) = std(I_f,0,3);

