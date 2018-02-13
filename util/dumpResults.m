function dumpResults(cnnSConfig, pred, layers_outputs, theta, numLayers)
%
% dumpResults dump the weights and output spikes of each layer
%
% Parameters:
%  cnnSConfig -  the config of the SNN, defined in the main directory.
%  pred - output the pred or not true: testing phrase, false: training phrase
%  layers_outputs -  the cell to store the outputs of each layer
%  layers_outputs.after - output spikes, the size is determined by layer
%  
%  theta      -  the cell to store W and b
%  theta{l}.W -  the weight in the l_th layer
%  theta{l}.b -  the bias in the l_th layer
%  numLayers  -  the number of the total layers
%%======================================================================
assert(numLayers == size(theta, 1));

for l = 2 : numLayers
    tempLayer = cnnSConfig.layer{l};
    tempTheta = theta{l};
    switch tempLayer.type
        case 'convspiking'
            if pred == false 
                filename = sprintf('%s_info', tempLayer.name);
                dumpWeights('weights/', filename, tempTheta.W);
                spike_path = sprintf('spikes/%s/train', tempLayer.name);
                dumpSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            else
                filename = sprintf('%s_info_trained', tempLayer.name);
                dumpWeights('weights/', filename, tempTheta.W);
                spike_path = sprintf('spikes/%s/test', tempLayer.name);
                dumpSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            end
        case 'poolspiking'
            if pred == false 
                spike_path = sprintf('spikes/%s/train', tempLayer.name);
                dumpSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            else
                spike_path = sprintf('spikes/%s/test', tempLayer.name);
                dumpSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            end
        case 'spiking'
            if pred == false 
                filename = sprintf('%s_info', tempLayer.name);
                dumpTripletWeights('weights/', filename, tempTheta.W, cnnSConfig.layer{l-1}.name, tempLayer.name);
                spike_path = sprintf('spikes/%s/train', tempLayer.name);
                dumpSparseSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            else
                filename = sprintf('%s_info_trained', tempLayer.name);
                dumpTripletWeights('weights/', filename, tempTheta.W, cnnSConfig.layer{l-1}.name, tempLayer.name);
                spike_path = sprintf('spikes/%s/test', tempLayer.name);
                dumpSparseSpikes(spike_path, tempLayer.name, layers_outputs{l}.after);
            end
        case 'stack2linespiking'
            continue;
        otherwise
            fprintf('Unrecognized layer type: %s\n', tempLayer.type);
    end
end
end


