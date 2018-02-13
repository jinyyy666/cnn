function [spikes] = spikeTimeSim(responses, vth, print_en, W_lat)
%spikeTimeSim Returns the binary response matrix given by accumulated
%response from the inputs
%
% Parameters:
%  responses - the 3D tensor of accumulated responses
%           responses(row, col, endTime)
%  
%  vth -
%         the threshold of the spiking neuron
%  W_lat - the lateral connection matrix, outputSize * outputSize

% Returns:
%  spikes - binary matrix of resulted spikes
%                      spikes(imageRow, imageCol, endTime)
if ~exist('W_lat','var') 
    W_lat = [];
end

[imageRow, imageCol, endTime] = size(responses);
spikes = zeros(imageRow, imageCol, endTime);
tm = 64;
ts = 8;
const_t_ref = 2;
v = zeros(imageRow, imageCol);
ep = zeros(imageRow, imageCol); % track synaptic response
t_ref = zeros(imageRow, imageCol); % refractory period

for t = 1:endTime % this is because the GPU does not cover the last time point
    % 1. Leakage:
    v = v - v/tm;
    ep = ep - ep/ts;
    if(t == 1)
        continue;
    end
    % 2. Add up the response to ep
    ep = ep + responses(:,:,t-1);
    if ~isempty(W_lat)
        tmp_lat = W_lat * reshape(permute(spikes(:,:,t-1), [2,1,3]), [], 1);
        tmp_lat = reshape(tmp_lat, imageCol, imageRow);
        ep = ep + tmp_lat';
    end
    
    
    % 3. Update the vmem accordingly (first order response)
    v = v + ep/ts;
    if print_en && imageRow > 74
        fprintf('@time %d: EP = %f, v = %f\n', t, ep(74,1), v(74,1));
    end
    % Reset the v if the t_ref > 0
    v(t_ref > 0) = 0;

    % Decrease the t_ref
    t_ref(t_ref > 0) = t_ref(t_ref > 0) - 1;
    
    % 4. See fire or not
    o_spike = zeros(imageRow, imageCol);
    o_spike(v > vth) = 1;
    spikes(:,:,t) = o_spike;
    
    % 5. Set the t_ref for the fired neurons
    t_ref(v > vth) = const_t_ref;
    v(v > vth) = 0;
end
end