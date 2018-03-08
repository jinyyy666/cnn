function pool_delta = backpropDeltaConv(conv_delta, input_spikes, output_spikes, weights)
% backpropDeltaConv:
%  back-propagate the delta from the convspiking layer to poolspiking
% 
% Parameters:
%  conv_delta  - the deltas of the convspiking layer
%                conv_deta(curDim, curDim)
%
%  input_spikes   - the 3D tensor of binary spikes in the poolspiking layer
%                 input_spikes(preDim, preDim, endTime)
%
%  output_spikes  - the 3D tensor of binary spikes in the convspiking layer
%                 pool_spikes(curDim, curDim, endTime)
%
%  weights - the weights of the convolution filters
%               weights(kernelSize, kernelSize)
%
% Returns:
%  pool_delta - back-proped delta from the convspiking
%               pool_delta(preDim, preDim)
%
[preDim, preDim_Y, endTime] = size(input_spikes);
assert(preDim == preDim_Y);
[curDim, curDim_Y, endTime_cur] = size(output_spikes);
assert(curDim == curDim_Y);
assert(endTime == endTime_cur);

[kernelSize, kernelSize_Y] = size(weights);
assert(kernelSize == kernelSize_Y);

pool_delta = zeros(preDim, preDim);
for i = 1:preDim
    for j = 1:preDim
        % implement the convolution of the deltas by two loops
        for x = 1:kernelSize
            for y = 1:kernelSize
                cx = i - x + 1;
                cy = j - y + 1;
                if (cx < 1 || cx > curDim || cy < 1 || cy > curDim)
                    continue
                end
                i_cnt = sum(input_spikes(i,j,:));
                o_cnt = sum(output_spikes(cx, cy, :));
                e = accumulateEffect(output_spikes(cx, cy,:), input_spikes(i,j,:));
                ratio = 1;
                if (i_cnt ~= 0 && o_cnt ~= 0)
                    ratio = e / i_cnt;
                end
                pool_delta(i,j) = pool_delta(i,j) + conv_delta(cx, cy) * weights(x,y) * ratio;
            end
        end
    end
end


end