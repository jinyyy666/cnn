function [cost, grad, preds] = cnnSpikingCost(theta, images, labels, cnnConfig, meta, pred)
% Calcualte cost and gradient for a spiking CNN
%                            
% Parameters:
%  theta      -  a vector parameter
%  images     -  stores images in imageDim x imageDim x endTime x channel x numImges
%                array    
%  labels     -  the labels for the output layer
%  pred       -  boolean only forward propagate and return
%                predictions
%
% Returns:
%  cost       -  cross entropy/mse cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end;



theta = thetaChangeSpiking(theta,meta,'vec2stack',cnnConfig);
%%======================================================================
%% STEP 1a: Forward Propagation
numLayers = size(theta, 1);
numImages = size(images,5);
layersizes = meta.layersize;



temp = cell(numLayers, 1);
grad = cell(numLayers, 1);
temp{1}.after = images;
%assert(isequal(size(images),[layersizes{1} numImages]),'layersize do not match at the input layer');

for l = 2 : numLayers
    tempLayer = cnnConfig.layer{l};
    tempTheta = theta{l};
    switch tempLayer.type
        case 'convspiking'
            [temp{l}.after] = cnnConvSpiking(temp{l-1}.after, tempTheta.W, [], tempLayer.vth);
        case 'poolspiking'
            [temp{l}.after, temp{l}.weights] = cnnPoolSpiking(tempLayer.poolDim, temp{l-1}.after, tempLayer.vth);
        case 'stack2linespiking'
            temp{l}.after = stack2lineSpiking(temp{l-1}.after);
        case 'spiking'
            if strcmp(tempLayer.name, 'output')
                temp{l}.after = Spiking(temp{l-1}.after, tempTheta.W, [], tempLayer.vth, tempLayer.W_lat);
            else
                temp{l}.after = Spiking(temp{l-1}.after, tempTheta.W, [], tempLayer.vth);
            end
    end
    %assert(isequal(size(temp{l}.after),[layersizes{l} numImages]),'layersize do not match at layer %d\n',l);
end

if cnnConfig.dump
    dumpResults(cnnConfig, pred, temp, theta, numLayers);
end

%%======================================================================
%% STEP 1b: Calculate Cost
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(sum(temp{numLayers}.after, 2),[],1);
    preds = preds';
    cost = 0;
    grad = 0;
    return;
end;

switch cnnConfig.costFun
    case 'crossEntropy'
        numClasses = cnnConfig.layer{numLayers}.dimension;
        extLabels = zeros(numClasses, numImages);
        extLabels(sub2ind(size(extLabels), labels', 1 : numImages)) = 1;
        cost = - mean(sum(extLabels .* log(temp{numLayers}.after)));
    case 'mse'
        numClasses = cnnConfig.layer{numLayers}.dimension;
        desired_level = cnnConfig.desired_level;
        undesired_level = cnnConfig.undesired_level;
        margin = cnnConfig.margin;
        extLabels = undesired_level * ones(numClasses, numImages);
        extLabels(sub2ind(size(extLabels), labels', 1 : numImages)) = desired_level;
        diff = sum(temp{numLayers}.after, 2) - extLabels;
        diff(abs(diff) <= margin) = 0;
        cost = sum(sum(diff.*diff));
end

%%======================================================================
%% STEP 1c: Backpropagation
%  modify the output spikes of the desired neurons if neccessary

if strcmp(cnnConfig.costFun, 'crossEntropy') && strcmp(tempLayer.type, 'softmax')
    temp{l}.gradBefore = temp{l}.after - extLabels;
    grad{l}.W = temp{l}.gradBefore * temp{l - 1}.after' / numImages;
    grad{l}.b = mean(temp{l}.gradBefore, 2);
elseif strcmp(cnnConfig.costFun, 'mse') && strcmp(tempLayer.name, 'output')
    temp{l}.lateral_factors = getLateralFactors(temp{l}.after,tempLayer, labels);
    temp{l}.after = modifyOutputSpikes(temp{l}.after, labels, cnnConfig.desired_level);
    temp{l}.gradBefore = diff / cnnConfig.layer{l}.vth;
    [temp{l}.accEffect, temp{l}.effectRatio] = synapticEffect(temp{l}.after, temp{l-1}.after, cnnConfig.use_effect_ratio);
    grad{l}.W = getGradW(temp{l}.accEffect, temp{l}.gradBefore, temp{l}.lateral_factors) + getGradWReg(theta{l}.W, cnnConfig);
    grad{l}.b = zeros(size(temp{l}.gradBefore, 1), 1);
end
assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l);

for l = numLayers-1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type
        case 'spiking'
            temp{l}.gradBefore = (theta{l + 1}.W .* temp{l+1}.effectRatio)' * temp{l + 1}.gradBefore / cnnConfig.layer{l}.vth;
            [temp{l}.accEffect, temp{l}.effectRatio] = synapticEffect(temp{l}.after, temp{l-1}.after, cnnConfig.use_effect_ratio);
            grad{l}.W = getGradW(temp{l}.accEffect, temp{l}.gradBefore) + getGradWReg(theta{l}.W, cnnConfig);
            grad{l}.b = zeros(size(temp{l}.gradBefore, 1), 1);
        case 'stack2linespiking'
            size_pool = size(temp{l - 1}.after);
            size_pool = [size_pool([1,2,4]), numImages];
            temp{l}.gradBefore = reshape((theta{l + 1}.W .* temp{l+1}.effectRatio)' * temp{l + 1}.gradBefore, size_pool);
            temp{l}.gradBefore = permute(temp{l}.gradBefore, [2,1,3,4]);
            grad{l}.W = [];
            grad{l}.b = [];
            break;
    end
    assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
    assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l);
end

for l = l - 1 : -1 : 2
    tempLayer = cnnConfig.layer{l};
    switch tempLayer.type 
        case 'poolspiking'
            size_pool = size(temp{l}.after);
            size_pool = [size_pool([1,2,4]), numImages];
            numChannel = size(temp{l}.after,4);
            temp{l}.gradAfter = zeros(size_pool);
            if isempty(theta{l + 1}.W)
                % the upper layer is 'stack2linespiking'
                temp{l}.gradAfter = temp{l + 1}.gradBefore;
            else
                % the upper layer is 'convspiking'
                for i = 1 : numImages
                    for c = 1 : numChannel
                        for j = 1 : size(temp{l + 1}.gradBefore, 3)
                            if cnnConfig.layer{l + 1}.conMatrix(c,j) ~= 0
                                temp{l}.gradAfter(:,:,c,i) = temp{l}.gradAfter(:,:,c,i) + conv2(temp{l + 1}.gradBefore(:,:,j,i), theta{l + 1}.W(:,:,c,j), 'full');
                            end
                        end
                    end
                end
            end
            size_conv = size(temp{l - 1}.after);
            size_conv = [size_conv([1,2,4]),numImages];
            temp{l}.gradBefore = zeros(size_conv);
            for i = 1 : numImages
                for c = 1 : numChannel
                    temp{l}.gradBefore(:,:,c,i) = kron(temp{l}.gradAfter(:,:,c,i), ones(tempLayer.poolDim)) .* temp{l}.weights(:,:,c,i);
                end
            end
            grad{l}.W = [];
            grad{l}.b = [];
        case 'convspiking'
            temp{l}.gradBefore = temp{l+1}.gradBefore / cnnConfig.layer{l}.vth;
            tempW = zeros([size(theta{l}.W) numImages]); 
            numInputMap = size(tempW, 3);
            numOutputMap = size(tempW, 4);
            for i = 1 : numImages
                for nI = 1 : numInputMap
                    for nO = 1 : numOutputMap
                        if tempLayer.conMatrix(nI,nO) ~= 0
                            tempW(:,:,nI,nO,i) = getGradWConv(temp{l}.after(:,:,:,nO,i), temp{l - 1}.after(:,:,:,nI,i), temp{l}.gradBefore(:,:,nO,i), size(theta{l}.W, 1), nI, nO);
                        end
                    end
                end
            end
            grad{l}.W = mean(tempW,5) + getGradWRegConv(theta{l}.W, cnnConfig);
%             if numInputMap == 1 && numOutputMap >= 3
%                                        %   row, col, input_channel, filter_id 
%                 fprintf('Wgrad: %f\n', grad{l}.W(5,3,1,5));
%             end
            grad{l}.b = zeros(numOutputMap,1);
                             
        otherwise 
            printf('%s layer is not supported', tempLayer.type);       
    end
    assert(isequal(size(grad{l}.W),size(theta{l}.W)),'size of layer %d .W do not match',l);
    assert(isequal(size(grad{l}.b),size(theta{l}.b)),'size of layer %d .b do not match',l); 
end

%% Unroll gradient into grad vector for minFunc
grad = thetaChange(grad,meta,'stack2vec',cnnConfig);

end
