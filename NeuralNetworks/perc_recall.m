function ret = perc_recall(p,x)
inp = p * [x; ones(1, size(x,2))];
ret = (inp >= 0);
end