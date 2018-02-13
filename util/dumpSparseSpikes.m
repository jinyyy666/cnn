function spikes = dumpSparseSpikes(path, file, spikes)
% Dump the spike timings in terms of the sparse format
% The sparse format:
%   -1           -1
%   [neuron_id] [spike_time]
%   ...         ...
%   -1          -1    
%
% The line "-1  -1" is used to separate spike times obtained under
% different iteration
%
% Parameters:
%  path - the path to dump the spikes
%  file - the file name        
%  spikes - the binary matrix of the spikes(outputSize, endTime)
%
%  Notice that the GPU simulator uses 0-based index but Matlab uses 1-based
%  index, we need to modify the dumped spike time
%
    if(path(end) ~= '/')
        path = [path, '/'];
    end
    if(~exist(path, 'dir'))
        mkdir(path)
    end

    [neuron_ids, times] = find(spikes > 0);
    
    %  do this because the GPU simulator is simulating from t = 0 - t =
    %  endTime - 1; But the matlab is simulating from t = 1 - t = endTime
    times = times - 1;
    neuron_ids = neuron_ids - 1;
    sparse = sortrows([neuron_ids, times]);
    sparse = [-1,-1 ;sparse;-1,-1];
    filename = sprintf('%s%s.dat', path, file);
    dlmwrite(filename, sparse, 'delimiter', '\t');
end