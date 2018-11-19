function [delta,s] = CrossVal(Name1,Name1L,Par1,Name2,Name2L,Par2,Pat,DOut,k,NoShuffle)
% Performs k-cross fold validation on given data and learning algorithms.
%
% inputs:
%   Name1   first algorithm
%   Name1L  first hypothesis
%   Par1    params of first algorithm
%   Name2   second algorithm
%   Name2L  second hypothesis
%   Par2    params of second algorithm
%   Pat     input set
%   DOut    desired output set
%   k       main param k-cross fold validation
%   NoShuffle (can be omitted), name is self explanatory
%           
% output:
%   delta   difference of errors
%   s       standard deviation of the estimator
if (nargin == 9)
% random shuffle
    Pat = Pat(:, randperm(size(Pat, 2)));
end
deltaI = [];
setSize = size(DOut, 2)/k;
for i=1:k
    % testing set
    Ti = Pat(:, (i-1)*setSize + 1:i*setSize);
    DTi = DOut(:, (i-1)*setSize + 1:i*setSize);
    % training set
    Si = Pat;
    Si(:, (i-1)*setSize + 1:i*setSize) = [];
    DSi = DOut;
    DSi(:, (i-1)*setSize + 1:i*setSize) = [];
    % learn!
    LPar1 = feval(Name1, Si, DSi, Par1);
    LPar2 = feval(Name2, Si, DSi, Par2);
    % compute errors
    errTiH1 = Err(Name1, Name1L, LPar1,Si, DSi, Ti, DTi);
    errTiH2 = Err(Name2, Name2L, LPar2, Si, DSi, Ti, DTi);
    deltaI(i)=errTiH1 - errTiH2;
end
delta = sum(deltaI)/k;
diff = (deltaI - delta).^2;
s = sqrt(1/(k*(k-1))*sum(diff));
end