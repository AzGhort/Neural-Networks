function E = Err(Name,NameL,Par,Tr,DTr,Ts,DTs)
% Computes error of learning algorithm.
%
% inputs:
%   Name  name of learning algorithm
%   NameL learned function
%   Par   params of learning algorithm
%   Tr    training set
%   DTr   desired out for training set
%   Ts    test set
%   DTs   desired out for test set
% output:
%   E     r/n, where r=number of missclassified points from test set
%         and n=size of test set
hyp = feval(Name,Tr,DTr,Par);
actualOut = feval(NameL, hyp, Ts);
numErrs = (actualOut ~= DTs);
E = sum(numErrs)/size(Ts,2);
end
