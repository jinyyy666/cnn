function [accEffect, effectRatio] = synapticEffect(outputs, inputs, use_effect_ratio)
% synapticEffect:
%  Compute the accumulative synaptic effect used for wgrad
%  Compute the effect ratio used for estimating f'(o^k_j) is applicable
% 
% Parameters:
%  outputs - the 3D tensor of binary spikes
%           outputs(outputSize, endTime, numImages)
%  inputs  - the 3D tensor of binary spikes
%           inputs(inputSize, endTime, numImages)
%
%  use_effect_ratio - use the effect ratio or not
%
% Returns:
%  accEffect - the accumulative synaptic effect, accEffect(outputSize, inputSize, numImages)
%  effectRatio - the effect ratio * W, effectRatio(outputSize, inputSize) 
%
[outputSize, ~, numImages] = size(outputs);
[inputSize, ~, numImages_I] = size(inputs);
assert(numImages == numImages_I);
accEffect = zeros(outputSize, inputSize, numImages);
effectRatio = ones(outputSize, inputSize);
for i = 1:numImages
    for j = 1:outputSize
        for k = 1:inputSize
            accEffect(j,k,i) = accumulateEffect(outputs(j,:,i), inputs(k,:,i));
        end
    end
end

if use_effect_ratio
    if numImages > 1
        fprintf('Warning::Disable the use_effect_ratio when minibatch > 1!!\n')
        return;
    end
    output_fireCount = sum(squeeze(outputs), 2);
    input_fireCount_mat = ones(outputSize, 1) * sum(squeeze(inputs), 2)';
    effects = accEffect(:,:,1);
    effectRatio = effects ./input_fireCount_mat;
    effectRatio(~isfinite(effectRatio)) = 1;
    effectRatio(output_fireCount == 0,:) = 1;
end


end