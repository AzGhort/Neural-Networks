function ret=perc_create(n)
ret=[];
for i=1:n
    ret=[ret;rand()];
end
ret=[ret;-rand()];
end