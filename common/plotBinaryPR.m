function [AP] = plotBinaryPR( SORTED )
GT = SORTED(:,2) > 0;
NS = size(GT,1);
SEQUENCE = zeros(NS+1,2);
PRED_POS_MASK = true(NS,1);
SAME = sum(PRED_POS_MASK .* GT);
P = SAME/sum(PRED_POS_MASK);
R = SAME/sum(GT);
SEQUENCE(1,:) = [P R];
for i = 1:NS
    PRED_POS_MASK(i) = false;
    SAME = sum(PRED_POS_MASK .* GT);
    P = SAME/sum(PRED_POS_MASK);
    if isnan(P),P=0;end % for when i==NS
    R = SAME/sum(GT);
    SEQUENCE(i+1,:) = [P R];
end
%figure;
%clf;
plot(SEQUENCE(:,2),SEQUENCE(:,1));

Ps = SEQUENCE(:,1);
Rs = SEQUENCE(:,2);


AP=0;
for t=0:0.1:1
    p=max(Ps(Rs>=t));
    if isempty(p)
        p=0;
    end
    AP=AP+p/11;
end


end



