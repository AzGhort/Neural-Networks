function ret = selectk(data,k)
    x = randperm(size(data,2),k);
    ret = data(:,x);
end