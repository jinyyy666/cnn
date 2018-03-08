function conv_delta = backpropDeltaPool(pool_delta, conv_spikes, pool_spikes, weights, pool_size)
% backpropDeltaPool:
%  back-propagate the delta from the poolspiking layer to convspiking
% 
% Parameters:
%  pool_delta  - the deltas of the poolspiking layer
%                pool_deta(poolDim, poolDim)
%
%  conv_spikes - the 3D tensor of binary spikes
%               conv_spikes(convDim, convDim, endTime)
%
%  pool_spikes  - the 3D tensor of binary spikes
%               pool_spikes(poolDim, poolDim, endTime)
%
%  weights - the weights of the propagated back deltas to the convspiking
%               weights(convDim, convDim)
%  pool_size -  the size of pooling
%               pool_size(size_x, size_y)
%
% Returns:
%  conv_delta - back-proped delta from the poolspiking 
%               conv_delta(convDim, convDim)
%
[convDim, convDim_Y, endTime] = size(conv_spikes);
assert(convDim == convDim_Y);
[poolDim, poolDim_Y, endTime_pool] = size(pool_spikes);
assert(poolDim == poolDim_Y);
assert(endTime == endTime_pool);
pool_row = pool_size(1);
pool_col = pool_size(2);

effectRatio = ones(convDim, convDim);
for i = 1:convDim
    p_x = floor((i - 1)/pool_row) + 1;
    for j = 1:convDim
        p_y = floor((j - 1)/pool_col) + 1;
        if(p_x > poolDim || p_y > poolDim)
            continue;
        end
        i_cnt = sum(conv_spikes(i, j, :));
        o_cnt = sum(pool_spikes(p_x, p_y, :));
        if(i_cnt ~= 0 && o_cnt ~= 0)
            effectRatio(i,j) = accumulateEffect(pool_spikes(p_x, p_y, :), conv_spikes(i, j, :))/i_cnt;
        end
    end
end
conv_delta = kron(pool_delta, ones(pool_size)) .* weights .* effectRatio;

end