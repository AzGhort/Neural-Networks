In2 = csvread('In2.csv');
c = csvread('c2.csv');
% in order to obtain the same answers from MemorizerRecall,
% we reset random numbers generator
stream = RandStream.getGlobalStream;
reset(stream)

Par1 = {[1 1 1 1 -1], 1, 50};

[d,s] = CrossVal('PLearn','PRecall',Par1,'Memorizer','MemorizerRecall',0,In2,c,6,'NoShuffle')