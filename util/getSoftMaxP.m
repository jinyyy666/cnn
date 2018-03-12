function prob = getSoftMaxP(outputs)
% getSoftMaxP:
%  Get the softMax probability given the output spikes in the last layer
% 
% Parameters:
%  outputs : binary matrix, outputSize * endTime * numImages
%  deltas - the delta associated with the output filter
%           deltas(rowO, colO);
%  kernelSize - the size of the convolution kernel
%
% Returns:
%  prob -  the probability matrix: outputSize * numImages
%

fireCounts = squeeze(sum(outputs, 2));
[outputSize, ~] = size(fireCounts);

% minus by the max to reduce the overflow
fireCounts = fireCounts - ones(outputSize, 1) * max(fireCounts);
sum_p = sum(exp(fireCounts), 1);
prob = exp(fireCounts) ./ (ones(outputSize, 1) * sum_p);

end

