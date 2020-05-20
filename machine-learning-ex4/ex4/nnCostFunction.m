function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add bias column to X matrix
X = [ones(size(X, 1), 1) X];

% Compute z2 
z2 = Theta1 * X';
% Compute a2 with z2
a2 = sigmoid(z2);
% Transpose a2 to make a 5000x25 size matrix 
a2 = a2';
% Add bias column to a2 matrix
a2 = [ones(size(a2, 1), 1) a2];
% Transpose again to make 26x5000 matrix
a2 = a2';

% Compute z3
z3 = Theta2 * a2;
% Compute a3 with z3
a3 = sigmoid(z3);
h = a3;
% Matrix transpose to make 5000x10
h = h';

% Range vector from 1 to 10
vector_comp = 1:num_labels;
% Y zeros vector size m x num_labels in this case 5000x10
Y = zeros(m, num_labels);

% Loop throw y and compare with range vector (from 1 to 10). Result in a vector of 0's and 1's where the 1's correspond to the number. e.g 5 corresponds to [0 0 0 0 1 0 0 0 0 0]
for k = 1:m
  a = vector_comp == y(k, :);
  Y(k, :) = a;
end  

% Cost Function
J = (1/m) * sum(sum(((-Y).* log(h)) - ((1-Y).*log(1-h))));
% Regularization 
reg = (lambda/(2*m)) * ((sum(sum(Theta1(:, 2:end).^2))) + (sum(sum(Theta2(:, 2:end).^2))));
% Cost function plus regularization
J = J + reg;

% Compute sigmoid derivative with z2
g_2 = sigmoidGradient(z2);
% Transpose to add bias column
g_2 = g_2';
% Add bias unit column
g_2 = [ones(size(g_2, 1), 1) g_2];
% Transpose to original matrix size
g_2 = g_2';

% Compute delta 3 with Y (error for each activation output unit)
delta_3 = a3' - Y;
% Compute delta 2
delta_2 = (Theta2' * delta_3').*g_2;
% Discard bias unit column from delta 2
delta_2 = delta_2(2:end, :);

% Update Theta 2 gradient
Theta2_grad = Theta2_grad + (delta_3' * a2');

% Compute sigmoid derivative with z1(equals to X == activation input unit)
g_1 = sigmoidGradient(X);
% Compute delta 1
delta_1 = (Theta1' * delta_2)'.*g_1;
% Discard bias unit column from delta 1
delta_1 = delta_1(:, 2:end);

% Update Theta 1 gradient
Theta1_grad = Theta1_grad + (delta_2 * X);

% Divide the accumulated gradients by 1/m (unregularized for the bias columns)
Theta1_grad(:, 1) = Theta1_grad(:, 1) / m;
Theta2_grad(:, 1) = Theta2_grad(:, 1) / m;

% Divide the accumulated gradients by 1/m (regularized with lambda and m)
Theta1_grad(:, 2:end) = (Theta1_grad(:, 2:end) / m) + ((lambda/m)*Theta1(:, 2:end));
Theta2_grad(:, 2:end) = (Theta2_grad(:, 2:end) / m) + ((lambda/m)*Theta2(:, 2:end));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
