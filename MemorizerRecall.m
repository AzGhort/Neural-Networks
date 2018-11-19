function Out = MemorizerRecall(LPar,In)
% A function simulating Memorizer
%
%     Out = MemorizerRecall(LPar,In)
%
% inputs:
%   Lpar  a cell array with training samples and
%         the respective desired outputs
%   In    a matrix of input vectors (as columns)
% output:
%   Out   a row vector; whenever the i-th input vector is
%         contained within the memorized input samples,
%         Out(i) equals the i-th remembered desired output,
%         otherwise it will be randomly 0 or 1
Known = LPar{1};
KnownOut = LPar{2};
j=1;
Out = zeros(1,size(In,2));
for i = 1:size(In,2)
    j=1;
    while j<=size(Known,2)
        if In(:,i)==Known(:,j)
            break
        else
            j=j+1;
        end
    end
    if j<=size(Known,2)
        Out(i) = KnownOut(j);
    else
        Out(i) = rand(1)>0.5;
    end
end
end