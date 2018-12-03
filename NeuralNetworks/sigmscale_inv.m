function scaled = sigmscale_inv(vector, lambda)
% inverse to sigmscale
for I=1:length(vector) 
    scaled(I)=(-1/lambda)*(log((1/vector(I))-1));
end
end