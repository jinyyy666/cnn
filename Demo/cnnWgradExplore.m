%% Investigate the grad for Spiking Convolution Neural Network

%% STEP 0: Initialize Parameters and Load Data
%  complete the configSpiking.m to config the network structure;

cnnSConfig = configSpiking();

% set up the paths
addpath('../');
addpath('../layer');
addpath('../TrainingMethod');
addpath('../Testing');
addpath('../DebugTools');
addpath('../util');
addpath('../Dataset/MNIST');

[theta, meta] = cnnSpikingInitParams(cnnSConfig);

%  initialize the weights from the checkPoint of the GPU
if isfield(cnnSConfig, 'ini_from_GPU') && cnnSConfig.ini_from_GPU == true
    theta = cnnSpikingInitParamsFromCheck(meta, cnnSConfig, '../checkPoint_cnn_current_best.txt');
    %theta = cnnSpikingInitParamsFromCheck(meta, cnnSConfig, '../checkPoint.txt');
end

% Load MNIST Data
if cnnSConfig.dump
    images = loadSpikes('../Dataset/MNIST/input_spikes_0_5.dat', cnnSConfig.layer{1}.dimension(1), meta.endTime);
    d = cnnSConfig.layer{1}.dimension;
    images = reshape(images,d(1),d(2),d(3),d(4),[]);
    labels = 6;
else
    d = cnnSConfig.layer{1}.dimension;
    images = loadSpikingMNISTImages('train-images-idx3-ubyte', d, cnnSConfig.train_samples);
    labels = loadSpikingMNISTLabels('train-labels-idx1-ubyte', cnnSConfig.train_samples);
    labels = labels + 1; % matlab uses 1-based index
    fprintf('Loading the training sample... Done!\n');
end

pred = false;

gradCheck(@(x, y, z) cnnSpikingCost(x, y, z, cnnSConfig, meta), theta, 1, images, labels);