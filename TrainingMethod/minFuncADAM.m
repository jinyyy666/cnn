function [opttheta] = minFuncADAM(funObj,theta,data,labels,...
                        options)
% Runs Adam to optimize the parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x endTime x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  minibatch*  - size of minibatch
%  alpha      - initial learning rate, default to be 0.001
%  b1     - beta1, controls the decay of first moment, default to be 0.9
%  b2     - beta2, controls the decay of second moment, default to be 0.999
%  eps    - parameter for making update robust, default to be 1e-8
%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','minibatch'})),...
        'Need to define both `epochs` and `minibatch`');
if ~isfield(options, 'alpha')
    options.alpha = 0.001;
end
if ~isfield(options,'b1')
    options.b1 = 0.9;
end;
if ~isfield(options, 'b2')
    options.b2 = 0.999;
end
if ~isfield(options, 'eps')
    options.eps = 1e-8;
end
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
b1 = options.b1;
b2 = options.b2;
eps = options.eps;
b1_t = b1;
b2_t = b2;
m = length(labels); % training set size
% Setup for momentum

g1 = zeros(size(theta));
g2 = zeros(size(theta));
%%======================================================================
%% Adam loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % get next randomly selected minibatch
        mb_data = data(:,:,:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost, grad] = funObj(theta,mb_data,mb_labels);
        
        % Estimate the first and second moment here:
        g1 = b1 * g1 + (1 - b1) * grad;
        g2 = b2 * g2 + (1 - b2) * grad .* grad;
        theta = theta - alpha * (g1/(1 - b1_t)) ./ (sqrt(g2/ (1 - b2_t)) + eps);
        b1_t = b1_t * b1;
        b2_t = b2_t * b2;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;

    % aneal learning rate by the sqrt
    alpha = alpha/sqrt(it+1);

end;

opttheta = theta;

end
