function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute hyphotheses 
H = X * theta;

% Cost function with regularitazion
J = ((1/(2*m)) * sum((H-y) .* (H-y))) + ((lambda/(2*m)) * sum(theta(2:end, :).*theta(2:end, :)));  

% Gradient descent to get the optimal theta parameters
% Gradient descent to first parameter with no regularitazion
grad(1:end, :) = (1/m) * sum(((H-y).*X(:,1:end)));
% Gradient descent to the rest of paramenters theta with regularitazion
grad(2:end, :) = grad(2:end, :) + ((lambda/m)*theta(2:end, :));

% =========================================================================

grad = grad(:);

end