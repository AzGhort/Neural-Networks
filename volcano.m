[X,Y] = meshgrid(0:.05:2, 0:.05:2);
Z=(X.*Y.*exp(-X.^2-Y.^2)) - 0.1*(X.*Y.*exp(-X.^16-Y.^16));
surf(X,Y,Z);