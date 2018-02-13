function [spikes_new] = modifyOutputSpikes(spikes, labels, desired_level)
% modifyOutputSpikes:
%  1. Modify the output spikes for the targeted neuron with zero spike cnt
%  2. For the target neuron (coresponding to the label), the fire count
%  after change should be the desired_level
% 
%
% Parameters:
%  spikes - the 3D tensor of binary spikes
%           spikes(outputSize, endTime, numImages)
%  
%  label - the label matrix: label(numClasses, numImages);
%  desired_level - the desired firing level of the output neuron
%
% Returns:
%  spikes_new - binary matrix of modified spikes
%                      spikes(outputSize, endTime, numImages)
spikes_new = spikes;
[outputSize, endTime, numImages] = size(spikes);
assert(size(labels, 1) == numImages);
for i = 1 : numImages
    label = labels(i);
    assert(label <= outputSize && label >= 1);
    interval = floor(endTime / desired_level);
    dummy = zeros(1, endTime);
    % the matlab start time 1, corresponding to GPU/CPU start time 0
    % In order to match with GPU, I move all the spikes by time 1
    dummy(interval+1:interval:endTime) = 1;
    spikes_new(label,:,i) = dummy;
end
end