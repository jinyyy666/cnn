%% Spiking Convolution Neural Network

% diary on;
% diary('run_log');

%% STEP 0: Initialize Parameters and Load Data
%  complete the configSpiking.m to config the network structure;

cnnSConfig = configSpiking_mnist();

% set up the paths
addpath('../');
addpath('../layer');
addpath('../TrainingMethod');
addpath('../Testing');
addpath('../DebugTools');
addpath('../util');
addpath('../Dataset/MNIST');

%  calling cnnInitParams() to initialize parameters
[theta, meta] = cnnSpikingInitParams(cnnSConfig);

%  initialize the weights from the checkPoint of the GPU
if isfield(cnnSConfig, 'ini_from_GPU') && cnnSConfig.ini_from_GPU == true
    theta = cnnSpikingInitParamsFromCheck(meta, cnnSConfig, '../checkPoint_mnist_best.txt');
end

% Load MNIST Data
if cnnSConfig.dump
    images = loadSpikes('../Dataset/MNIST/input_spikes_0_5.dat', cnnSConfig.layer{1}.dimension(1), meta.endTime);
    d = cnnSConfig.layer{1}.dimension;
    images = reshape(images,d(1),d(2),d(3),d(4),[]);
    labels = 6;
else
    d = cnnSConfig.layer{1}.dimension;
    %images = loadSpikingMNISTImages('train-images-idx3-ubyte', d, cnnSConfig.train_samples);
    %labels = loadSpikingMNISTLabels('train-labels-idx1-ubyte', cnnSConfig.train_samples);
    images = loadSpikingMNISTImages('t10k-images-idx3-ubyte', d, cnnSConfig.train_samples);
    labels = loadSpikingMNISTLabels('t10k-labels-idx1-ubyte', cnnSConfig.train_samples);
    labels = labels + 1; % matlab uses 1-based index
    fprintf('Loading the training sample... Done!\n');
end
%%======================================================================
%% STEP 3: Learn Parameters
%  Select 1) SGD or 2) Adam to train the network.
options.epochs = 1;
if ~isfield(cnnSConfig, 'minibatch')
    cnnSConfig.minibatch = 1;
end
options.minibatch = cnnSConfig.minibatch;
switch cnnSConfig.optimizer
    case 'SGD'
        options.alpha = 1e-5;
        options.momentum = .95;
        opttheta = minFuncSGD(@(x,y,z) cnnSpikingCost(x,y,z,cnnSConfig,meta),theta,images,labels,options);
    case 'ADAM'
        options.alpha = 0.001;
        opttheta = minFuncADAM(@(x,y,z) cnnSpikingCost(x,y,z,cnnSConfig,meta),theta,images,labels,options);
    otherwise
        fprintf('Unrecognized optimizer type: %s', cnnSConfig.optimizer);
end

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

if cnnSConfig.dump
    testImages = loadSpikes('../Dataset/MNIST/input_spikes_0_5.dat', cnnSConfig.layer{1}.dimension(1), meta.endTime);
    testLabels = 6;
else
    d = cnnSConfig.layer{1}.dimension;
    testImages = loadSpikingMNISTImages('t10k-images-idx3-ubyte', d, cnnSConfig.test_samples);
    testLabels = loadSpikingMNISTLabels('t10k-labels-idx1-ubyte', cnnSConfig.test_samples);
    testLabels = testLabels + 1; % matlab uses 1-based index
end


[cost,grad,preds]=cnnSpikingCost(opttheta,testImages,testLabels,cnnSConfig,meta,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

%diary off;
