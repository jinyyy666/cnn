function [sideEffect] = getGradWConvSideEffect(outputs, inputs, ws, kernelSize, vth, numInputMap, numOutputMap)
% getGradWConvSideEffect:
%  Compute the side effect factor for each output neuron
% 
% Parameters:
%  outputs - the 5D tensor of binary spikes
%           outputs(rowO, colO, endTime, numOutputMap, numImages)
%  inputs  - the 5D tensor of binary spikes
%           outputs(rowI, colI, endTime, numInputMap, numImages)
%  ws      - the 4D tensor of the weights for kernels
%           ws(kernelSize, kernelSize, numInputMap, numOutputMap)
%  kernelSize - the size of the convolution kernel
%
% Returns:
%  sideEffect - the side effect factors
%           sideEffect(rowO, colO, numOutputMap, numImages);
%
[rowO, colO, endTime, numOutputMap_o, numImages] = size(outputs);
[rowI, colI, endTime_I, numInputMap_i, ~] = size(inputs);
assert(endTime == endTime_I);
assert(numOutputMap_o == numOutputMap);
assert(numInputMap_i == numInputMap);

sideEffect = zeros(rowO, colO, numOutputMap, numImages);

for im = 1:numImages
    for oo = 1:numOutputMap
        for x = 1:rowO
            for y = 1:colO
                val = 0;
                for ii = 1:numInputMap
                for i = 1:kernelSize
                    cx = i + x - 1;
                    for j = 1:kernelSize
                        cy = j + y - 1;
                        if(cx >= 1 && cy >= 1 && cx <= rowI && cy <= colI)
                            e = accumulateEffect(outputs(x,y,:,oo,im), inputs(cx,cy,:,ii,im));
                            o_cnt = sum(outputs(x,y,:,oo,im));
                            ratio = 0.5;
                            if(o_cnt ~= 0)
                                ratio = e/o_cnt;
                            end
                            val = val + ratio * ws(i, j, ii, oo);
                        end
                    end
                end
                end
                sideEffect(x, y, oo, im) = val;
            end
        end
    end
end
sideEffect = sideEffect / vth;

end