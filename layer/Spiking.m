function [output] = Spiking(input,W, b, vth, W_lat)
% Spiking computes output spikes of the input spikes and the weights
% 
% Parameters:
%  input : matrix, inputSize * endTime
%  W     : matrix, outputSize * inputSize
%  b     : vector, outputSize * 1
%  W_lat : the lateral connection matrix, outputSize * outputSize
% 
% Returns:
%  output : matrix, OutputSize * endTime
if ~exist('W_lat','var') 
    W_lat = [];
end
% reshape the input into (inputSize, 1, endTime) so that we can use 
% the spiketimeSim
[inputSize, endTime] = size(input);
[outputSize, inputSize_W] = size(W);
assert(inputSize == inputSize_W);

input_response = W * input;
input_response_reshaped = reshape(input_response, outputSize, 1, endTime);

output = spikeTimeSim(input_response_reshaped, vth, false, W_lat);
output = reshape(output, outputSize, endTime);
end
