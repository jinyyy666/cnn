function [convolvedFeatures] = cnnConvSpiking(images, W, b, vth, con_matrix, shape)
%cnnConvSpiking Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  images - large images to convolve with, matrix in the form
%           images(row, col, channel, image number)
%  W, b - W, b 
%         W is of shape (filterDim,filterDim,channel,numFilters)
%         b is of shape (numFilters,1)
%  vth -
%         the threshold of the convolution spiking neuron
%  con_matrix -
%         the connection between input channel and output maps. If the ith
%         input channel has connection with jth output map, then
%         con_matrix(i,j) = 1, otherwise, 0;
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, endTime, featureNum, imageNum)

[filterDimRow,filterDimCol,channel,numFilters] = size(W);

if ~exist('con_matrix','var') || isempty(con_matrix)
    con_matrix = ones(channel, numFilters);
end

if ~exist('shape','var')
    shape = 'valid';
end

[imageDimRow, imageDimCol, endTime, ~, numImages] = size(images);
convDimRow = imageDimRow - filterDimRow + 1;
convDimCol = imageDimCol - filterDimCol + 1;

convolvedFeatures = zeros(convDimRow, convDimCol, endTime, numFilters, numImages);

%   Convolve every filter with every image here to produce the convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, endTime, featureNum, imageNum)

for imageNum = 1:numImages
  for filterNum = 1:numFilters
      convolvedResponse = zeros(convDimRow, convDimCol, endTime); 
      for channelNum = 1:channel
          if con_matrix(channelNum,filterNum) ~= 0
            % Obtain the feature (filterDim x filterDim) needed during the convolution
            filter = W(:,:,channelNum,filterNum); 

            % Flip the feature matrix because of the definition of convolution, as explained later
            filter = rot90(squeeze(filter),2);

            % Obtain the image
            im = squeeze(images(:, :, :, channelNum,imageNum));

            % Convolve "filter" with "im", adding the result to
            % convolvedResponse be sure to do a 'valid' convolution
            % Notice that the response from the final time point will not
            % be considered
            for t = 1:endTime
                convolvedResponse(:,:,t) = convolvedResponse(:,:,t) + conv2(im(:,:,t), filter, shape);
            end
          end
      end
      % Perform the spike timing simulation, convert the input responses to
      % spikes
      convolvedImage = spikeTimeSim(convolvedResponse, vth, false);
      convolvedFeatures(:, :, :, filterNum, imageNum) = convolvedImage;
  end
end

end

