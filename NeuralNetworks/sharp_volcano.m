% base of volcano
BaseX = [0,8,8,0]; 
BaseY = [0,0,-8,-8];
BaseZ = [0,0,0,0];
fill3(BaseX, BaseY, BaseZ,'green'), hold on;

% sides of volcano
Side1X = [0,8,6,2];
Side1Y = [0,0,-2,-2];
Side1Z = [0,0,8,8];
fill3(Side1X, Side1Y, Side1Z,'green'), hold on;
Side2X = [0,8,6,2];
Side2Y = [-8,-8,-6,-6];
Side2Z = [0,0,8,8];
fill3(Side2X, Side2Y, Side2Z,'green'), hold on;
Side3X = [0,0,2,2];
Side3Y = [0,-8,-6,-2];
Side3Z = [0,0,8,8];
fill3(Side3X, Side3Y, Side3Z,'green'), hold on;
Side4X = [8,8,6,6];
Side4Y = [0,-8,-6,-2];
Side4Z = [0,0,8,8];
fill3(Side4X, Side4Y, Side4Z,'green'), hold on;

% sides of caldera
Caldera1X = [2,3,5,6];
Caldera1Y = [-2,-3,-3,-2];
Caldera1Z = [8,6,6,8];
fill3(Caldera1X, Caldera1Y, Caldera1Z,'red'), hold on;
Caldera2X = [2,3,5,6];
Caldera2Y = [-6,-5,-5,-6];
Caldera2Z = [8,6,6,8];
fill3(Caldera2X, Caldera2Y, Caldera2Z,'red'), hold on;
Caldera3X = [2,3,3,2];
Caldera3Y = [-2,-3,-5,-6];
Caldera3Z = [8,6,6,8];
fill3(Caldera3X, Caldera3Y, Caldera3Z,'red'), hold on;
Caldera4X = [6,5,5,6];
Caldera4Y = [-2,-3,-5,-6];
Caldera4Z = [8,6,6,8];
fill3(Caldera4X, Caldera4Y, Caldera4Z,'red'), hold on;
% caldera bottom
CalderaX = [3,5,5,3];
CalderaY = [-3,-3,-5,-5];
CalderaZ = [6,6,6,6];
fill3(CalderaX, CalderaY, CalderaZ,'red'), hold on;
