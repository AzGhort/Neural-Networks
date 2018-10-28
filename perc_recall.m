function ret = perc_recall(p,x)
s = p*[x;1];
if (s >= 0)
    ret = 1;
else
    ret = 0;
end
end