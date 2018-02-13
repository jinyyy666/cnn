%% Spiking Convolution Neural Network

% diary on;
% diary('run_log');

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

%  calling cnnInitParams() to initialize parameters
[theta, meta] = cnnSpikingInitParams(cnnSConfig);

% Load MNIST Data
%images = loadMNISTImages('train-images-idx3-ubyte');
images = loadSpikes('../Dataset/MNIST/input_spikes_0_5.dat', cnnSConfig.layer{1}.dimension(1), meta.endTime);
d = cnnSConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),d(4),[]);
%labels = loadMNISTLabels('train-labels-idx1-ubyte');
%labels(labels==0) = 10; % Remap 0 to 10
labels = 6;
%%======================================================================
%% STEP 3: Learn Parameters
%  Select 1) SGD or 2) Adam to train the network.
options.epochs = 1;
options.minibatch = 1;
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

%testImages = loadMNISTImages('t10k-images-idx3-ubyte');
%testImages = reshape(testImages,d(1),d(2),d(3),[]);
%testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
%testLabels(testLabels==0) = 10; % Remap 0 to 10
testImages = loadSpikes('../Dataset/MNIST/input_spikes_0_5.dat', cnnSConfig.layer{1}.dimension(1), meta.endTime);
testLabels = 6;

[cost,grad,preds]=cnnSpikingCost(opttheta,testImages,testLabels,cnnSConfig,meta,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

%diary off;
