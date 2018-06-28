function sideEffect = getGradWSideEffect(accEffect, outputs, weights, vth)
% getGradWSideEffect:
%  Compute the side effect for each weight given the accumulative synaptic
%  effects and t
% 
% Parameters:
%  accEffect - the accumulative synaptic effect, 
%              accEffect(outputSize, inputSize, numImages)
%  outputs - the 3D tensor of binary spikes
%           outputs(outputSize, endTime, numImages)
%  weights - the 2D tensor of the weights
%           weights(outputSize, inputSize);
% 
%
% Returns:
%  sideEffect - the sideEffect for the weight
%           sideEffect(outputSize, numImages);
%
[outputSize, inputSize, numImages] = size(accEffect);
sideEffect = zeros(outputSize, numImages);

for i = 1:numImages
    output_fireCount = sum(squeeze(outputs(:,:,i)), 2);
    output_fireCount_mat = output_fireCount * ones(1, inputSize);
    e_partial_o = accEffect(:,:,i) ./ output_fireCount_mat;
    e_partial_o(~isfinite(e_partial_o)) = 0.5;
    e_partial_o = weights .* e_partial_o;
    sideEffect(:, i) = sum(e_partial_o, 2);
end
sideEffect = sideEffect / vth;


end