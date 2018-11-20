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
if ~exist('NoShuffle','var')
     % random shuffle
     randVector = randperm(size(Pat, 2));
     Pat = Pat(:,randVector);
     DOut = DOut(:,randVector);
end
deltaI = zeros(1,5);
setSize = size(DOut, 2)/k;
for i=1:k
    % testing set
    Ti = Pat(:, (i-1)*setSize + 1:i*setSize);
    DTi = DOut(:, (i-1)*setSize + 1:i*setSize);
    % training set
    Si = [Pat(:,1:(i-1)*setSize), Pat(:,i*setSize + 1: k*setSize)];
    DSi = [DOut(:,1:(i-1)*setSize), DOut(:,i*setSize + 1: k*setSize)];
    % compute errors
    errTiH1 = Err(Name1, Name1L, Par1,Si, DSi, Ti, DTi);
    errTiH2 = Err(Name2, Name2L, Par2, Si, DSi, Ti, DTi);
    deltaI(i)=errTiH1 - errTiH2;
end
delta = sum(deltaI)/k;
s = std(deltaI);
end