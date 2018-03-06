function images = loadSpikingMNISTImages(filename, dimension, num_samples)
%--------------------------------------------------------------------------
%loadSpikingMNISTImages returns a 4D tensor 28 x 28 x endTime x nChannels x [number of MNIST images] 
% that contains the MNIST images in binary spikes
%
% Params:
%   dimension: the dimensin of the input layer (nRows, nCols, endTime, nChannels)
%   num_samples: the number of samples needed for training/testing
%--------------------------------------------------------------------------
ims = loadMNISTImages(filename); % #pixels x #examples, already rescale to [0,1]
ims = ims(:, 1:num_samples);
ims = reshape(ims, dimension(1), dimension(2), num_samples);
% change the image to the row major order
endTime = dimension(3);
images = zeros(dimension(1), dimension(2), endTime, 1, num_samples);
for i = 1:num_samples
    images(:,:,:,:,i) = repmat(ims(:,:,i), [1,1,endTime]);
end

% matlab starts from t = 1, which corresponds to the t = 0 for CPU/GPU
images(:,:,1,1,:) = 0;
images = images / 5.5;
rands = rand(size(images));

images(images > rands) = 1;
images(images <= rands) = 0;

end