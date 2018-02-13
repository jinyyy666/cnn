function [pooledFeatures, weights] = cnnPoolSpiking(poolDim, convolvedFeatures, vth)
%cnnPooling Pools the given convolved features in terms of spike trains
%
% Parameters:
%  poolDim - dimension of pooling region, is a 1 * 2 vector(poolDimRow poolDimCol);
%  convolvedFeatures - convolved features to pool (as given by cnnConvSpiking)
%                      convolvedFeatures(imageRow, imageCol, endTime, featureNum, imageNum)
%  vth -    the threshold of the pooling spiking neuron
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, endTime, featureNum, imageNum)
%  weights        - how much the input contributes to the output
%     



numImages = size(convolvedFeatures, 5);
numFilters = size(convolvedFeatures, 4);
endTime = size(convolvedFeatures, 3);
convolvedDimRow = size(convolvedFeatures, 1);
convolvedDimCol = size(convolvedFeatures, 2);
pooledDimRow = floor(convolvedDimRow / poolDim(1));
pooledDimCol = floor(convolvedDimCol / poolDim(2));

weights = zeros(convolvedDimRow, convolvedDimCol, numFilters, numImages);
featuresTrim = convolvedFeatures(1:pooledDimRow*poolDim(1),1:pooledDimCol*poolDim(2),:,:,:);

fT_size = [pooledDimRow*poolDim(1), pooledDimCol*poolDim(2), numFilters, numImages];
weights(1:pooledDimRow*poolDim(1), 1:pooledDimCol*poolDim(2),:,:) = ones(fT_size) / poolDim(1) / poolDim(2);

pooledFeatures = zeros(pooledDimRow, pooledDimCol, endTime, numFilters, numImages);

poolFilter = ones(poolDim);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        features = featuresTrim(:,:,:, filterNum, imageNum);
        pooledResponse = zeros(pooledDimRow, pooledDimCol, endTime);
        for t = 1:endTime
            poolConvolvedFields = conv2(features(:,:,t), poolFilter, 'valid');
            pooledResponse(:,:,t) = poolConvolvedFields(1:poolDim:end, 1:poolDim:end);
        end
        pooledImage = spikeTimeSim(pooledResponse, vth, false);
        pooledFeatures(:,:,:,filterNum, imageNum) = pooledImage;
    end
end

end
 


