function visualize_weights(weights)
% config the sizes
config.layer{2}.size = [5, 5, 16]; % conv1
config.layer{3}.size = [5, 5, 40]; % conv2
config.layer{4}.size = [100, 640]; % hidden
config.layer{5}.size = [100, 10]; % output

figure
%im = zeros(4*5, 4*5);
for j = 1:config.layer{3}.size(3)
    %x = floor(j/4) + 1;
    %y = mod(j-1, 4) + 1;
    subplot(5,8,j);
    mat = reshape(weights(400+((j-1)*25 + 1) : 400 + j*25), [5, 5])';
    imshow(mat, []);
    
    %im(x*(1:5),y*(1:5)) = reshape(weights((j-1)*25 + 1 : j*25), [5, 5])';
end
%imagesc(im);
end
