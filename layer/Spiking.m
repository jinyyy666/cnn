function [outputs] = Spiking(input,W, b, vth, W_lat)
% Spiking computes output spikes of the input spikes and the weights
% 
% Parameters:
%  input : matrix, inputSize * endTime * numImages
%  W     : matrix, outputSize * inputSize
%  b     : vector, outputSize * 1
%  W_lat : the lateral connection matrix, outputSize * outputSize
% 
% Returns:
%  outputs : matrix, outputSize * endTime * numImages
if ~exist('W_lat','var') 
    W_lat = [];
end
% reshape the input into (inputSize, 1, endTime) so that we can use 
% the spiketimeSim
[inputSize, endTime, numImages] = size(input);
[outputSize, inputSize_W] = size(W);
assert(inputSize == inputSize_W);
outputs = zeros(outputSize, endTime, numImages);

for imageNum = 1:numImages
    input_response = W * input(:,:,imageNum);
    input_response_reshaped = reshape(input_response, outputSize, 1, endTime);

    output = spikeTimeSim(input_response_reshaped, vth, false, W_lat);
    output = reshape(output, outputSize, endTime);
    outputs(:,:,imageNum) = output;
end

end
