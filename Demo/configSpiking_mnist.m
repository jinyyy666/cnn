function cnnConfigSpiking = configSpiking_mnist()

l = 1;
cnnConfigSpiking.layer{l}.type = 'input';
cnnConfigSpiking.layer{l}.dimension = [28 28 400 1];
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'stack2linespiking';
cnnConfigSpiking.layer{l}.name = 'stack2line';
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'spiking';
cnnConfigSpiking.layer{l}.name = 'hidden';
cnnConfigSpiking.layer{l}.dimension = 800;
cnnConfigSpiking.layer{l}.vth = 20;
l = l + 1;

cnnConfigSpiking.layer{l}.type = 'spiking';
cnnConfigSpiking.layer{l}.name = 'output';
cnnConfigSpiking.layer{l}.dimension = 10;
cnnConfigSpiking.layer{l}.vth = 8;
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
cnnConfigSpiking.lambda = 0;
cnnConfigSpiking.beta = 0;
cnnConfigSpiking.weight_limit = 8;

% dump the simulation result
cnnConfigSpiking.dump = false;
% initialize from the GPU
cnnConfigSpiking.ini_from_GPU = true;
% visualize the result
cnnConfigSpiking.visualize = true;
end