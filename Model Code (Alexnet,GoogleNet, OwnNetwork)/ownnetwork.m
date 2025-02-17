% Load the tea leaf images and labels
imds = imageDatastore('test1', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% Set up the data augmentation for the training dataset
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1], ...
    'RandXShear', [-10, 10], ...
    'RandYShear', [-10, 10], ...
    'FillValue', 0);

augimdsTrain = augmentedImageDatastore([224, 224, 3], imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsVal = augmentedImageDatastore([224, 224, 3], imdsVal);

% Define the model architecture
layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(1024)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

% Set the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsVal, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain, layers, options);
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
