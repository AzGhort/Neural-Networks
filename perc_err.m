function ret = perc_err(p,x,c)
results = perc_recall(p,x) ~= c
ret = sum(results)/size(x,2);
end