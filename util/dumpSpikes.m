function dumpSpikes( path, file, spikes)
%dumpSpikes dump the given spikes matrix into the given path
% Here we dump the entire binary spike matrix instead of the spike times
%
% Parameters:
%  path - the path to dump the spikes
%  file - the file name        
%  spikes - the binary matrix of the spikes(row, col, endTime, featureNum, imageNum)
    if(path(end) ~= '/')
        path = [path, '/'];
    end
    if(~exist(path, 'dir'))
        mkdir(path)
    end
    [row, col, endTime, featureNum, numImages] = size(spikes);
    for imageNum = 1:numImages
        filename = sprintf('%s%s_%d.dat', path, file, imageNum);
        for filterNum = 1:featureNum
            mat = squeeze(spikes(:,:,:,filterNum, imageNum));
            % Do not need to handle the spike time matchs here because we
            % are dump the entire binary spike matrix!
            
            % permute to make the dump in the row-first order
            mat_per = permute(mat, [2, 1, 3]);
            % mat_vec is in the format of [endTime] * [row*col]
            mat_vec = reshape(mat_per, 1, endTime*row*col);
            if(filterNum == 1)
                dlmwrite(filename, mat_vec, 'delimiter', ' ');
            else
                dlmwrite(filename, mat_vec, '-append', 'delimiter', ' ');
            end
        end
    end
end

