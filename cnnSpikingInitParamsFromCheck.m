function theta = cnnSpikingInitParamsFromCheck(meta, cnnSConfig, filename)
% Initialize parameters from the checkPoint dumped by GPU
%                            
% Parameters:
%  meta       -  meta param 
%       numTotalParams : total number of the parameters
%       numParams      : the number of the parameters each layer
%       numLayers      : the total number of layers
%  cnnSConfig -  the config struct
%  filename     : the filename for the dumped check point
%
% Returns:
%  theta      -  parameter vector
%  
oldTheta = load(filename);
% get rid of the lateral inihibition
if isfield(cnnSConfig.layer{end}, 'local_ini')
    last_l_dim = cnnSConfig.layer{end}.dimension;
    num_lat = last_l_dim * last_l_dim;
    oldTheta(end - num_lat -  last_l_dim + 1 : end - last_l_dim) = [];
end
total_params = meta.numTotalParams;
assert(total_params == length(oldTheta), 'Mismatch between the Config between Matlab and GPU!');
% Since the matlab'default reshape is col-major but our GPU dump in a
% row-major order, we need to re-org the theta vec into the form that is
% col-major order

newTheta = cell(meta.numLayers,1);
for i = 1 : meta.numLayers
    tempLayer = cnnSConfig.layer{i};
    switch tempLayer.type
        case 'input'
            newTheta{i}.W = [];
            newTheta{i}.b = [];
            row = tempLayer.dimension(1);
            col = tempLayer.dimension(2);
            channel = tempLayer.dimension(4);
        case 'convspiking'
            row = row + 1 - tempLayer.filterDim(1);
            col = col + 1 - tempLayer.filterDim(2);          
            newTheta{i}.W = reshape(oldTheta(1:meta.numParams(i,1)),meta.paramsize{i});
            newTheta{i}.W = permute(newTheta{i}.W, [2,1,3,4]);
            oldTheta(1:meta.numParams(i,1))=[];
            channel = tempLayer.numFilters;
            newTheta{i}.b = oldTheta(1:channel);
            oldTheta(1:channel) = [];
        case 'poolspiking'
            newTheta{i}.W = [];
            newTheta{i}.b = [];
            row = int32(row/tempLayer.poolDim(1));
            col = int32(col/tempLayer.poolDim(2));
        case 'stack2linespiking'
            newTheta{i}.W = [];
            newTheta{i}.b = [];
            row = row * col * channel;
            col = 1;
            channel = 1;
        case 'spiking'
            paramsize =[meta.paramsize{i}(2), meta.paramsize{i}(1)];
            newTheta{i}.W = reshape(oldTheta(1:meta.numParams(i,1)),paramsize);
            newTheta{i}.W = permute(newTheta{i}.W, [2,1]);
            oldTheta(1:meta.numParams(i,1)) = [];
            dimension = tempLayer.dimension;
            newTheta{i}.b = oldTheta(1:dimension);
            oldTheta(1:dimension) = [];
    end
end
theta = thetaChangeSpiking(newTheta, meta, 'stack2vec', cnnSConfig);    
end
