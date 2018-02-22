function [wgrad] = getGradWConv(outputs, inputs, deltas, kernelSize, nI, nO)
% getGradWConv:
%  Compute the wgrad for each convolution filter
% 
% Parameters:
%  outputs - the 3D tensor of binary spikes
%           outputs(rowO, colO, endTime)
%  inputs  - the 3D tensor of binary spikes
%           inputs(rowI, colI, endTime)
%  deltas - the delta associated with the output filter
%           deltas(rowO, colO);
%  kernelSize - the size of the convolution kernel
%
% Returns:
%  wgrad - the weight gradient
%           wgrad(kernelSize, kernelSize);
%
[rowO, colO, endTime] = size(outputs);
[rowI, colI, endTime_I] = size(inputs);
assert(endTime == endTime_I);
[rowO_d, colO_d] = size(deltas);
assert(rowO == rowO_d && colO == colO_d);
wgrad = zeros(kernelSize);
for i = 1:kernelSize
    for j = 1:kernelSize
        val = 0;
        for x = 1:rowO
            cx = i + x - 1;
            for y = 1:colO
                cy = j + y - 1;
                if(cx >= 1 && cy >= 1 && cx <= rowI && cy <= colI)
                    e = accumulateEffect(outputs(x,y,:), inputs(cx,cy,:));
                    val = val + e * deltas(x, y);
%                     if(i == 5&& j == 3&& nI == 1 && nO == 5)
%                         fprintf('Collect x = %d; y = %d; Acc effect: %f\tdelta = %f\n', x, y, e, deltas(x, y));
%                     end
                end
            end
        end
        wgrad(i,j) = val;
    end
end

end