function fea = scaletest(fea, type, minvalue, maxvalue)
% normalize features. This step empirically improves the performance
% written by Liefeng Bo on 01/04/2011 in University of Washington

switch type
  case 'linear'
	for i = 1:size(fea,2)
	    fea(:,i) = (fea(:,i) - minvalue)./(maxvalue - minvalue);
	end
  case 'power'
        ppp = 0.3;
        for i = 1:size(fea,2)
            fea(:,i) = sign(fea(:,i)).*abs(fea(:,i)).^ppp;
        end
  otherwise
       disp('Unknown type');
end

