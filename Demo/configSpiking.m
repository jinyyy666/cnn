function cnnConfigSpiking = configSpiking()

l = 1;
cnnConfigSpiking.layer{l}.type = 'input';
cnnConfigSpiking.layer{l}.dimension = [28 28 450 1];
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'convspiking';
cnnConfigSpiking.layer{l}.name = 'conv1';
cnnConfigSpiking.layer{l}.filterDim = [5 5];
cnnConfigSpiking.layer{l}.numFilters = 16;
cnnConfigSpiking.layer{l}.conMatrix = ones(1,16);
cnnConfigSpiking.layer{l}.vth = 5;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'poolspiking';
cnnConfigSpiking.layer{l}.name = 'pool1';
cnnConfigSpiking.layer{l}.poolDim = [2 2];
cnnConfigSpiking.layer{l}.vth = 2;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'convspiking';
cnnConfigSpiking.layer{l}.name = 'conv2';
cnnConfigSpiking.layer{l}.filterDim = [5 5];
cnnConfigSpiking.layer{l}.numFilters = 40;
cnnConfigSpiking.layer{l}.conMatrix = ones(16,40);
cnnConfigSpiking.layer{l}.vth = 10;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'poolspiking';
cnnConfigSpiking.layer{l}.name = 'pool2';
cnnConfigSpiking.layer{l}.poolDim = [2 2];
cnnConfigSpiking.layer{l}.vth = 4;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'stack2linespiking';
cnnConfigSpiking.layer{l}.name = 'stack2line';
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'spiking';
cnnConfigSpiking.layer{l}.name = 'hidden';
cnnConfigSpiking.layer{l}.dimension = 300;
cnnConfigSpiking.layer{l}.vth = 10;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'spiking';
cnnConfigSpiking.layer{l}.name = 'output';
cnnConfigSpiking.layer{l}.dimension = 10;
cnnConfigSpiking.layer{l}.vth = 5;
% Add the local inihibition
cnnConfigSpiking.layer{l}.local_ini = 1;
W_lat = -1*cnnConfigSpiking.layer{l}.local_ini * ones(10);
for i = 1:10
    W_lat(i,i) = 0;
end
cnnConfigSpiking.layer{l}.W_lat = W_lat;
l = l + 1;

% number of training/testing samples
cnnConfigSpiking.train_samples = 1;
cnnConfigSpiking.test_samples = 1;

% mini-batch size
cnnConfigSpiking.minibatch = 1;

% other params:
cnnConfigSpiking.costFun = 'mse';
cnnConfigSpiking.desired_level = 35;
cnnConfigSpiking.undesired_level = 5;
cnnConfigSpiking.margin = 5;
cnnConfigSpiking.use_effect_ratio = true;
cnnConfigSpiking.optimizer = 'ADAM';

% regularization params
cnnConfigSpiking.lambda = 10;
cnnConfigSpiking.beta = 0.4;
cnnConfigSpiking.weight_limit = 8;

% dump the simulation result
cnnConfigSpiking.dump = false;

% visualize the activations and variances
cnnConfigSpiking.visualize = true;

% load the weight from GPU
cnnConfigSpiking.ini_from_GPU = true;
end