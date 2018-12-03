function ret = perc_learn(p,x,c,lam,maxit)
iter = 0;
n = p;
while iter < maxit
    er = perc_err(n,x,c);
    if (er == 0)
        break;
    end
    n = perc_update(n,x,c,lam);
    iter=iter+1;
end
ret = n;
end