function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


%%%%5
%% Compute and choose best parameters (C and sigma) with cross validation error using a range of values [0.01, 0.03, 0.1, (...)]
%%%%%
%x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
%sim = gaussianKernel(x1, x2, sigma);

%vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%possi = [vals; vals];
%possi = possi';
%final = [0 0 0];
%for i=1:size(possi, 1)
%  for j=1:size(possi, 1)
%    C = possi(i, 1);
%    sigma = possi(j, 2);
%    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    predictions = svmPredict(model, Xval);
%    error = mean(double(predictions ~= yval));
%    temp = [C sigma error];
%    final = [final; temp];
%  endfor
%endfor  

%final;


%% Best paramenters after looping throw values
C = 1 
sigma = 0.1

% =========================================================================

end
