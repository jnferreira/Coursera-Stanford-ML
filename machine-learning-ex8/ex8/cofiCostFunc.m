function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Compute Cost Function
M = ((Theta*X')' - Y).^2;
% Compute final Cost Function with summation and with only R(i, j) = 1 or Cost function with movies with rating
cs = sum(sum(R.*M));
% Regularized Theta
reg_theta = (lambda * sum(sum(Theta.^2))) / 2;
% Regularized X
reg_X = (lambda * sum(sum(X.^2))) / 2;
% Divide cost function and Final result
J = (cs/2) + reg_theta + reg_X;


% Loop throw number of movies 
for i = 1:size(R, 1)
  % Only movies with attributed rating
  idx = find(R(i, :) == 1);
  % Filter Theta like the formula (Filter Theta so that have only movies rated by other users)
  Theta_temp = Theta(idx, :);
  % Filter Y like the formula (Filter with only users that have rated movies)
  Y_temp = Y(i, idx);
  % Regularization
  reg_X_grad = (lambda * X(i, :));
  % Upgrade gradient X 
  X_grad(i, :) += (X(i, :) * Theta_temp' - Y_temp) * Theta_temp + reg_X_grad;
endfor

% Loop throw number of users 
for j = 1:size(R, 2)
  % Only users with attributed rating
  idx = find(R(:, j) == 1);
  % Filter X like the formula (Filter X so that have only users with rated movies)
  X_temp = X(idx, :);
  % Filter Y like the formula (Filter with only movies that have been rated by users)
  Y_temp = Y(idx, j);
  % Regularization
  reg_Theta_grad = (lambda * Theta(j, :));
  % Upgrade gradient Theta 
  Theta_grad(j, :) += (X_temp * Theta(j, :)' - Y_temp)' * X_temp + reg_Theta_grad;
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
