function labels = loadSpikingMNISTLabels(filename, num_samples)
%--------------------------------------------------------------------------
%loadMNISTLabels returns a [num_samples]x1 matrix containing
%the labels for the MNIST images
%
% Params:
%   num_samples: the number of samples needed for training/testing
%--------------------------------------------------------------------------
labels = loadMNISTLabels(filename);
labels = labels(1:num_samples);
end
