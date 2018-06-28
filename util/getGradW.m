function wgrad = getGradW(accEffect, sideEffect, deltas, lateralFactors)
% getGradW:
%  Compute the gradient for the weight given the accumulative synaptic
%  effects and the lateralFactors
% 
% Parameters:
%  accEffect - the accumulative synaptic effect, 
%              accEffect(outputSize, inputSize, numImages)
%  sideEffect - for modelling the side effect of changing the weight
%              sideEffect(outputSize, numImages)
%  deltas - the error in the current layer
%               deltas (outputSize, numImages);
%  lateralFactors - the factor introduced by the local inhibition
%           lateralFactors(outputSize, numImages);
% 
%
% Returns:
%  wgrad - the gradient for the weight
%           wgrad(outputSize, inputSize);
%
[outputSize, inputSize, numImages] = size(accEffect);
[outputSize_s, numImages_s] = size(sideEffect);
[outputSize_D, numImages_D] = size(deltas);
assert(outputSize_D == outputSize);
assert(numImages_D == numImages);
assert(outputSize_s == outputSize && numImages_s == numImages);

if ~exist('lateralFactors', 'var') || isempty(lateralFactors)
    lateralFactors= ones(outputSize, numImages);
end

wgrad = zeros(outputSize, inputSize);

for i = 1:numImages
    lFactors = lateralFactors(:,i) * ones(1, inputSize);
    ds = deltas(:,i) * ones(1, inputSize);
    s_effect = sideEffect(:,i) * ones(1, inputSize);
    wgrad = wgrad + ds.* lFactors .* accEffect(:,:,i) .* (1 + s_effect);
end
wgrad = wgrad / numImages;


end