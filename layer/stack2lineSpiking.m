function [linedFeatures] = stack2lineSpiking(stackedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  stackedFeatures -   stacked features (as given by cnnpoolSpiking)
%                      stackedFeatures(row, col, endTime, featureNum, numImages)
%
% Returns:
%  linedFeatures  - combine the spike trains which belong to the same featureNum
%                   linedFeatures(row*col*featureNum, endTime, numImages)
%     

[row, col, endTime, featureNum, numImages] = size(stackedFeatures);
linedFeatures = zeros(row*col*featureNum, endTime, numImages);
for imageNum = 1:numImages
    stackedFeatures_permuted = permute(stackedFeatures(:,:,:,:,imageNum), [2,1,4,3]);
    linedFeatures(:,:,imageNum) = reshape(stackedFeatures_permuted, [], endTime);
end

end

