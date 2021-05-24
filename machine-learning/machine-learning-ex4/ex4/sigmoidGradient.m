function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).



%{
//I wrote this code before reading the pdf. lol
//it's the same thing as g(1-g)
skin = 1e-4;
peniswithskin = sigmoid(z+skin); 
%suck my zick plus skin LOL please let me have my laugh T_T 
peniswithnoskin = sigmoid(z-skin);
g = (peniswithskin-peniswithnoskin)/(2*skin);
%}

g=sigmoid(z).*(1-sigmoid(z));







% =============================================================




end
