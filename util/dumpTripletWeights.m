function dumpTripletWeights( path, file, weights, input_name, output_name)
%dumpWeights dump the given weight into the given path with a sparse format
% The sparse format is consistent with the CPU dump
% The format:
% [#index]  input_name_[#i_index]   output_name_[#o_idex]   [#weight_value]
%
% Parameters:
%  path - the path to dump the weights
%           
%  weights - the weight matrix of the weights(outputSize, inputSize)
    if(path(end) ~= '/')
        path = [path, '/'];
    end
    if(~exist(path, 'dir'))
        mkdir(path)
    end
    [outputSize, inputSize] = size(weights);
    filename = sprintf('%s%s.txt', path, file);
    fid = fopen(filename, 'w');
    for i = 0:inputSize-1
        for j = 0:outputSize-1
            input_str = sprintf('%s_%d', input_name, i);
            output_str = sprintf('%s_%d', output_name, j);
            w = weights(j+1, i+1);
            fprintf(fid, '%d\t%s\t%s\t%f\n', i*outputSize + j, input_str, output_str, w);
        end
    end
    fclose(fid);   
end
