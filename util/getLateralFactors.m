function lateral_factors = getLateralFactors(spikes, layer, labels)
% getLateralFactors:
%  Compute the factors associated with each neuron when introducing the
%  local inhibition
% 
% Parameters:
%  spikes - the 3D tensor of binary spikes
%           spikes(outputSize, endTime, numImages)
%  layer - the output layer config
%  layer.w0 - the strength of the local inhibition
%  layer.vth - the threshold of the output neuron
% 
%  labels - the label matrix: label(numClasses, numImages);
%
% Returns:
%  lateral_factors - the resulted lateral_factors for each output neuron
%                    lateral_factors(outputSize, numImages)
[outputSize, ~, numImages] = size(spikes);
assert(size(labels, 1) == numImages);
lateral_factors = ones(outputSize, numImages);
w0 = layer.local_ini;
vth = layer.vth;
if isempty(layer.W_lat)
    return;
end

for i = 1 : numImages
    label = labels(i);
    assert(label <= outputSize && label >= 1);
    for j = 1 : outputSize
        d_sum = 0;
        f_cnt_j = sum(spikes(j,:,i));
        d_j = 0;
        if(f_cnt_j >0 || (f_cnt_j == 0 && j == label))
            d_j = 1/vth;
        end
        for l = 1:outputSize
            if(l == j)
                continue;
            end
            f_cnt_l = sum(spikes(l,:,i));
            d_l = 0;
            if(f_cnt_l > 0 || (f_cnt_l == 0 && l == label))
                d_l = 1/vth;
            end
            % j ---> l
            e_jl = accumulateEffect(spikes(l,:,i), spikes(j,:,i));
            if(f_cnt_j == 0 || f_cnt_l == 0)
                effect_ratio_jl = 1;
            else
                effect_ratio_jl = e_jl / f_cnt_j;
            end
            % l ---> j
            e_lj = accumulateEffect(spikes(j,:,i), spikes(l,:,i));
            if(f_cnt_l == 0 || f_cnt_j == 0)
                effect_ratio_lj = 1;
            else
                effect_ratio_lj = e_lj / f_cnt_l;
            end
            
            d_sum  = d_sum + effect_ratio_jl * d_l * effect_ratio_lj * d_j;
        end
        lateral_factors(j, i) =  1/(1 - w0*w0*d_sum);
    end
end
end