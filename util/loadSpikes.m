function spikes = loadSpikes(filename, inputDim, end_time)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the spike timings from the dump file
% The file format:
%   -1           -1
%   [neuron_id] [spike_time]
%   ...         ...
%   -1          -1    
%
% The line "-1  -1" is used to separate spike times obtained under
% different iteration
%
% Input : 
%       filename : the file name
%       inputDim : the dim of the image, 28 for MNIST
%       endTime  : the end time of the spike train
% Output:
%       spikes : the matrix of the spike times (inputDim x inputDim x t)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(~exist(filename, 'file'))
        spikes = [];
        return;
    end
    data = load(filename);
    indices = find(data(:, 1) == -1);
    
    spikes = zeros(inputDim, inputDim, end_time);
    
    end_index = indices(end)-1; % read the hidden layer response
    
    if(end_index ~= 0)
        begin_index = indices(length(indices)-1)+1;
        data = data(begin_index : end_index,:);
        [row, ~] = size(data);
         
        for j = 1:row
            neuron_idx = data(j, 1);
            r = floor(neuron_idx / inputDim) + 1;
            c = mod(neuron_idx, inputDim) + 1;
            assert(r >= 1 & r <= inputDim);
            assert(c >= 1 & c <= inputDim);
            % Add 1 to the spike times because the matlab starts from 1
            spikes(r, c, data(j,2)+1) = 1;
        end

    end
end