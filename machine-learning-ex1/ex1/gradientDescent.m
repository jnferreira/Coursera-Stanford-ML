function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp_a = 0;
hyphoteses_a = 0;

temp_b = 0;
hyphoteses_b = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    temp_a = 0;
    hyphoteses_a = 0;
    temp1 = 0;
    
    for i = 1:m
      temp_a = ((theta(1, 1) * X(i, 1) + theta(2, 1) * X(i, 2)) - y(i, 1)) * X(i, 1);
      hyphoteses_a = hyphoteses_a + temp_a;
    end  
    
    Jtemp1 = alpha * (1/m) * hyphoteses_a;
    temp1 = theta(1,1) - Jtemp1;
    %theta(1,1) = theta(1,1) - Jtemp1
    
    temp_b = 0;
    hyphoteses_b = 0;
    temp2 = 0;
    
    for i = 1:m
      temp_b = ((theta(1, 1) * X(i, 1) + theta(2, 1) * X(i, 2)) - y(i, 1)) * X(i, 2);
      hyphoteses_b = hyphoteses_b + temp_b;
    end  
    
    Jtemp2 = alpha * (1/m) * hyphoteses_b;
    temp2 = theta(2,1) - Jtemp2;
    %theta(2,1) = theta(2,1) - Jtemp2
    
    theta(1,1) = temp1;
    theta(2,1) = temp2;
  

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end