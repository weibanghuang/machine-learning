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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
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
X=[ones(m,1), X];
yi=zeros(size(y, 1), num_labels); %5000x10
for t = 1:m
    yi(t, y(t))=1;
end

z2 = X*transpose(Theta1); %5000x25
a2 = sigmoid(z2); %5000x25
a2 = [ones(size(a2,1),1),a2]; %5000x26
z3 = a2*transpose(Theta2); %5000x10
h0x = sigmoid(z3); %5000x10

a=-yi.*log(h0x); %5000x10
b=(1-yi).*log(1-h0x); %5000x10
J = (1/m)*sum(sum(a-b));
Theta1_copy=Theta1;
Theta2_copy=Theta2;
Theta1_copy(:,1) = []; %remove first column 25x401->25x400
Theta2_copy(:,1) = []; %remove first column 10x26->10x25

reg = (lambda/(2*m))*(sum(sum(Theta1_copy.^2))+sum(sum(Theta2_copy.^2)));

J += reg;
% -------------------------------------------------------------

pussy = zeros(m,size(Theta2,1)); %5000x10

for t = 1:m

a1 = X(t,:); %1x401
z2 = a1*transpose(Theta1); %1x25
a2 = sigmoid(z2); %1x25
a2 = [1, a2]; %1x26
z3 = a2*transpose(Theta2); %1x10
a3 = sigmoid(z3); %1x10
pussy(t,:)=a3; %this is literally the same thing as the code above. I just forgot this line when I was writing the code above, and I was wonderin why my code wasnt working. smfh.
delta3 = a3 - yi(t, :);%1x10
delta2 = (delta3*Theta2_copy).*sigmoidGradient(z2); %1x25

Theta2_grad += transpose(delta3)*a2;
Theta1_grad += transpose(delta2)*a1;

end

Theta2_grad /= m;
Theta1_grad /= m;
% =========================================================================

%regularization
Theta1_grad(:,2:size(Theta1_grad,2)) += (lambda/m) * Theta1(:,2:size(Theta1,2));
Theta2_grad(:,2:size(Theta2_grad,2)) += (lambda/m) * Theta2(:,2:size(Theta2,2));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
