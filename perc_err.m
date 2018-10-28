function ret = perc_err(p,x,c)
ret = 0;
for i=1:size(x,2)
    cur = x(:, i);
    actual_output = perc_recall(p,cur);
    if (actual_output ~= c(i))
        ret = ret+1;
    end
end

end