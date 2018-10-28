function ret = perc_update(p,x,ideal_output,lam)
ret = p;
for i=1:size(x,2)
    cur = x(:, i);
    actual_output = perc_recall(p,cur);
    cur=([cur;0])';
    if (actual_output== 0 && ideal_output(i) == 1)
        ret=ret+lam*cur;
    elseif (actual_output == 1 && ideal_output(i) == 0)
        ret=ret-lam*cur;
    end
end
end