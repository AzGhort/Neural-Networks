function scaled = sigmscale(vector, lambda)
% scale data by sigmoid with steepness lambda
for I=1:length(vector) 
    scaled(I)=1/(1+exp(-vector(I)*lambda));
end
end