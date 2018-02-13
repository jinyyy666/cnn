function [theta, meta] = cnnSpikingInitParams(cnnSConfig)
% Initialize parameters
%                            
% Parameters:
%  cnnConfigSpiking    - spiking cnn configuration variable
%
% Returns:
%  theta      -  parameter vector
%  meta       -  meta param 
%       numTotalParams : total number of the parameters
%       numParams      : the number of the parameters each layer
%       numLayers      : the total number of layers
%
numLayers = size(cnnSConfig.layer,2);
theta = cell(numLayers,1);
numParams = zeros(numLayers,2);
meta.layersize = cell(numLayers,1);
meta.paramsize = cell(numLayers,1);
if cnnSConfig.dump
    % fix a random number so that each time the dumped spikes will be fixed
    rng(7);
end
for i = 1 : numLayers
    tempLayer = cnnSConfig.layer{i};
    
    switch tempLayer.type
        case 'input'
            theta{i}.W = [];
            theta{i}.b = [];
            row = tempLayer.dimension(1);
            col = tempLayer.dimension(2);
            endTime = tempLayer.dimension(3);
            channel = tempLayer.dimension(4);
            meta.layersize{i} = [row, col, endTime, channel];
        case 'convspiking'        
            row = row + 1 - tempLayer.filterDim(1);
            col = col + 1 - tempLayer.filterDim(2);          
            meta.paramsize{i} = [tempLayer.filterDim channel tempLayer.numFilters];
            theta{i}.W = randn(meta.paramsize{i});
            if cnnSConfig.dump
                % for verifying the GPU only
                theta{i}.W = round(theta{i}.W, 2);
            end
                             % [W, b]
            numParams(i,:) = [tempLayer.filterDim(1)*tempLayer.filterDim(2)*channel*tempLayer.numFilters tempLayer.numFilters];
            channel = tempLayer.numFilters;
            theta{i}.b = zeros(channel, 1);
            meta.layersize{i} = [row, col, endTime, channel];
        case 'poolspiking'
            theta{i}.W = [];
            theta{i}.b = [];
            row = int32(row/tempLayer.poolDim(1));
            col = int32(col/tempLayer.poolDim(2));
            meta.layersize{i} = [row, col, endTime, channel];
        case 'stack2linespiking'
            theta{i}.W = [];
            theta{i}.b = [];
            row = row * col * channel;
            col = 1;
            channel = 1;
            dimension = row;
            meta.layersize{i} = [dimension, endTime];
        case 'spiking'
            % initialisation of dnn method
            meta.paramsize{i} = [tempLayer.dimension dimension];
            %r = sqrt(6) ./ sqrt(double(dimension) + tempLayer.dimension);
            %theta{i}.W = rand(tempLayer.dimension, dimension) * 2 .* r - r;
            
            theta{i}.W = 2*rand(tempLayer.dimension, dimension) - 1;
            if cnnSConfig.dump
                % for verifying the GPU only
                theta{i}.W = round(theta{i}.W, 2);
            end
            numParams(i,:) = [tempLayer.dimension*dimension tempLayer.dimension];
            dimension = tempLayer.dimension;
            theta{i}.b = zeros(dimension, 1);
            meta.layersize{i} = [dimension, endTime];
            
    end
end
meta.numTotalParams = sum(sum(numParams));
meta.numParams = numParams;
meta.numLayers = numLayers;
meta.endTime = endTime;
theta = thetaChangeSpiking(theta, meta, 'stack2vec', cnnSConfig);
end
