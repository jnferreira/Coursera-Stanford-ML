function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add 1's row refering to bias unit
X = [ones(rows(X), 1) X];

% Compute z2 to use in sigmoid function
z2 = Theta1 * X';
% Compute activation units for hidden layer (2)
a2 = sigmoid(z2);

% Add 1's row refering to bias activation unit
a2 = [ones(rows(a2'), 1), a2'];

% Compute z3 to use in sigmoid function
z3 = Theta2 * a2';
% Compute hyphotheses for each class (Output layer)
h3 = sigmoid(z3);

h3 = h3';

[x, ix] = max(h3, [], 2);

for i=1:m
  p(i) = ix(i);
end 

% =========================================================================


end
