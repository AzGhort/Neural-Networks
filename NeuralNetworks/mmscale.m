function scaled = mmscale(vector,lower_bound,upper_bound)
% nmscale maps scales the input vector linearly into given interval
if lower_bound >= upper_bound
    error('Upper bound must be greater than lower bound');
end
max1 = max(vector);
min1 = min(vector);
output = zeros(size(vector));
a = (upper_bound-lower_bound)/(max1-min1);
b = upper_bound-a*max1;

for i = 1:length(vector)
    if (vector(i) == max1) output(i) = upper_bound;
    elseif (vector(i) == min1) output(i) = lower_bound;
    else output(i)=a*vector(i)+b;  
    end
end

scaled = output;
end