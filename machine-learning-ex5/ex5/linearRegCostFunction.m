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

% Recall that hypothesis h(x) is X*theta
% recall again that y is the actual output
% also recall that we do not regularize the first theta therefore we go 2:end

J = 1/(2*m)*sum((X*theta-y).^2)+lambda/(2*m)*sum(theta(2:end).^2);

% recalling that the initial theta is zero
newtheta = theta;
newtheta(1) = 0;

grad = (1/m)*sum((X*theta-y).*X)+(lambda/m)*newtheta';



% =========================================================================

grad = grad(:);

end
