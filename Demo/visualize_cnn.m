function visualize_cnn(acti, grad, cnnSConfig)
% Visualize the activations and gradient of each layer
%                            
% Parameters:
%  acti       -  the activation cells
%       after : the layer activation
%  grad       -  the gradient cells
%       W     -  the gradient for the weight
%  cnnSConfig -  the config struct
%
%
figure(1);
for l = 1:length(acti)
    tempLayer = cnnSConfig.layer{l};
    layer_type = tempLayer.type;
    if(strcmp(layer_type, 'spiking') || strcmp(layer_type, 'input'))
        a = squeeze(sum(acti{l}.after, 2));
        fprintf('The variance of the outputs in %d_th layer: %f\n', l, var(a(:)));
        subplot(1, length(acti), l);
        histogram(a, 'Normalization','probability');
        axis([0,120,0,1]);
    elseif(strcmp(layer_type, 'convspiking') || strcmp(layer_type, 'poolspiking'))
        a = squeeze(sum(acti{l}.after, 3));
        fprintf('The variance of the outputs in %d_th layer: %f\n', l, var(a(:)));
        subplot(1, length(acti), l);
        histogram(a, 'Normalization','probability');
        axis([0,120,0,1]);
    end
end

figure(2);
for l = 1:length(grad)
    if isfield(grad{l}, 'W') && ~isempty(grad{l}.W)
        subplot(1,length(grad),l);
        histogram(grad{l}.W(:), 'Normalization','probability');
        fprintf('The variance of the W gradient in %d_th layer: %f\n', l, var(grad{l}.W(:)));
    end
end

numImage = size(acti{1}.after, 5);
if(numImage == 1)
    for l = 1:length(acti)
        tempLayer = cnnSConfig.layer{l};
        switch tempLayer.type
            case 'input'
                figure(2+l);
                imshow(sum(acti{l}.after, 3), []);
            case 'convspiking'
                figure(2+l);
                n_filters = size(acti{l}.after, 4);
                a = sum(acti{l}.after, 3);
                for i = 1:n_filters
                    subplot(floor(n_filters/5)+1, 5, i);
                    imshow(a(:,:,i), []);
                end
            case 'poolspiking'
                figure(2+l);
                n_filters = size(acti{l}.after, 4);
                a = sum(acti{l}.after, 3);
                for i = 1:n_filters
                    subplot(floor(n_filters/5)+1, 5, i);
                    imshow(a(:,:,i), []);
                end
        end
    end
end
       
end

