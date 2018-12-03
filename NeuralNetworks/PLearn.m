function LPar = PLearn(x,c,Par)
    p = perc_learn(Par{1},x,c,Par{2},Par{3});
    LPar = {p};
end