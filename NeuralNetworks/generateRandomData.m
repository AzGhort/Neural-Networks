function grd = generateRandomData(m, n, S, R)
% creates MxN matrix of random values with normal distribution, EX=S, var=R
    grd = sqrt(R)*randn(m,n)+S;
end