function scaled = sdscale(vector)
% scale data so that they have EX=0 and var=1
EX = (1/length(vector))*sum(vector);
sdeviation = sqrt((sum((vector-EX).^2)/(length(vector)-1)));

scaled=(vector-EX)/sdeviation;
end