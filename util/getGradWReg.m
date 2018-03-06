function gradWReg = getGradWReg(W, cnnConfig)
% getGradWReg:
%  Compute the gradient for the regularization part of the spiking
%  layer
% 
% Parameters:
%  W - the 4D tensor of the weight
%           outputs(kernelSize, kernelSize, numInputMap, numOutputMap)
%  cnnConfig  - the config of the Spiking CNN
%  deltas - the delta associated with the output filter
%           deltas(rowO, colO);
%  kernelSize - the size of the convolution kernel
%
% Returns:
%  gradWReg - the weight gradient for the regularization part
%             gradWReg(kernelSize, kernelSize, numInputMap, numOutputMap);
%

% regularizatin params:
if ~isfield(cnnConfig, 'lambda')
    cnnConfig.lambda = 0;
end
if ~isfield(cnnConfig, 'beta')
    cnnConfig.beta = 0;
end
if ~isfield(cnnConfig, 'weight_limit')
    cnnConfig.weight_limit = 8;
end
lambda = cnnConfig.lambda;
beta = cnnConfig.beta;
weight_limit = cnnConfig.weight_limit;

[~, inputSize] = size(W);
W_norm = W .* W / (weight_limit * weight_limit);

sqSum = sum(W_norm, 2) / inputSize;



gradWReg = W / weight_limit;
gradWReg = lambda * beta * gradWReg .* exp(beta * (sqSum *ones(1, inputSize) - 1));
end


