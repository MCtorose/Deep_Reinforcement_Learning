inputSize = [256 256 3]; % 假设输入图片是 256x256 大小的彩色图片
numClasses = 2; % 假设我们有两个分类
lgraph = unetLayers(inputSize, numClasses);

% 显示网络结构
figure;
plot(lgraph);

predict_py = py.importlib.import_module('predict');



