%% Lung Cancer Detection using CNN in MATLAB
clc; clear; close all;

%% Load and Prepare Dataset
datasetPath = fullfile('Lung_cancer_dataset', 'Train');
testPath = fullfile('Lung_cancer_dataset', 'Test');

% Set Image Size
imageSize = [512 512 3];

% Create Image Datastore
trainDatastore = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

testDatastore = imageDatastore(testPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Check Class Distribution
disp('Training Data Class Distribution:');
countEachLabel(trainDatastore)

disp('Testing Data Class Distribution:');
countEachLabel(testDatastore)

% Split Data into Train & Validation Sets
[trainData, valData] = splitEachLabel(trainDatastore, 0.9, 'randomized');

%% Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1]);

augTrainData = augmentedImageDatastore(imageSize, trainData, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');
augValData = augmentedImageDatastore(imageSize, valData, ...
    'ColorPreprocessing', 'gray2rgb');

%% Sample Image Display
sampleImage = readimage(trainDatastore, 10);
imshow(sampleImage);
title('Sample Image');

%% Define CNN Model Architecture
layers = [
    imageInputLayer(imageSize, 'Normalization', 'zerocenter')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'WeightsInitializer', 'he')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    dropoutLayer(0.4)
    
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.3)
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    
    classificationLayer
];

%% Training Options (Improved)
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augValData, ...
    'ValidationPatience', 8, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'gpu');

%% Train the Model
[net, trainInfo] = trainNetwork(augTrainData, layers, options);

%% Save the Model and Training Info
save('lung_cancer_model.mat', 'net', 'trainInfo');

%% Test Model Performance
augTestData = augmentedImageDatastore(imageSize, testDatastore, ...
    'ColorPreprocessing', 'gray2rgb');
YPred = classify(net, augTestData);
YTest = testDatastore.Labels;

% Display Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');
