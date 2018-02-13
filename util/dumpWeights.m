function dumpWeights( path, file, weights)
%dumpWeights dump the given weight into the given path
%
% Parameters:
%  path - the path to dump the weights
%           
%  weights - the weight matrix of the weights(row, col, channel, featureNum)
    if(path(end) ~= '/')
        path = [path, '/'];
    end
    if(~exist(path, 'dir'))
        mkdir(path)
    end
    [row, col, channel, featureNum] = size(weights);
    filename = sprintf('%s%s.txt', path, file);
    mat = weights;
    mat_per = permute(mat, [2,1,3,4]);
    mat_vec = reshape(mat_per, 1, row*col*channel*featureNum);
    dlmwrite(filename, mat_vec, 'delimiter', ' ');    
end
