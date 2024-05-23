function lgraph = unetLayers(inputSize, numClasses)
% unetLayers Create a U-Net network
    
    encoderDepth = 4;
    filterSize = [3 3];
    numFilters = 64;
    maxPoolingSize = [2 2];
    
    layers = [
        imageInputLayer(inputSize, 'Name', 'input_layer', 'Normalization', 'none')
        ];
    
    % 编码器路径
    for i = 1:encoderDepth
        encoderBlock = [
            convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', ['encoder' num2str(i) '_conv1'])
            reluLayer('Name', ['encoder' num2str(i) '_relu1'])
            convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', ['encoder' num2str(i) '_conv2'])
            reluLayer('Name', ['encoder' num2str(i) '_relu2'])
            ];
        
        layers = [
            layers
            encoderBlock
            ];
        
        if i < encoderDepth
            layers = [
                layers
                maxPooling2dLayer(maxPoolingSize, 'Stride', 2, 'Name', ['encoder' num2str(i) '_maxpool'])
                ];
            
            numFilters = numFilters * 2;
        end
    end
    
    % 解码器路径
    for i = encoderDepth:-1:1
        numFilters = numFilters / 2;
        
        decoderBlock = [
            transposedConv2dLayer(2, numFilters, 'Stride', 2, 'Cropping', 'same', 'Name', ['decoder' num2str(i) '_upconv'])
            convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', ['decoder' num2str(i) '_conv1'])
            reluLayer('Name', ['decoder' num2str(i) '_relu1'])
            convolution2dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', ['decoder' num2str(i) '_conv2'])
            reluLayer('Name', ['decoder' num2str(i) '_relu2'])
            ];
        
        layers = [
            layers
            decoderBlock
            ];
    end
    
    finalLayers = [
        convolution2dLayer(1, numClasses, 'Padding', 'same', 'Name', 'final_conv')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'pixelLabels')
        ];
    
    lgraph = layerGraph(layers);
    lgraph = addLayers(lgraph, finalLayers);
    % 由于pixelClassificationLayer是输出层，因此不需要手动连接它
    
    % 这里可以添加跳过连接（如果需要），连接编码器层和对应的解码器层
    % 例如: lgraph = connectLayers(lgraph, 'encoder1_conv2', 'decoder4_upconv');
    
    % 显示网络结构
    figure;
    plot(lgraph);
end
