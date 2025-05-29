% Lung Cancer Classification with Tumor Size and Staging
clear; clc;

% Define Paths
baseDir = 'Lung_cancer_dataset'; % Adjust to absolute path if needed, e.g., 'C:/Data/Lung_cancer_dataset'
datasetPath = fullfile(baseDir, 'Train');
testPath = fullfile(baseDir, 'Test');

% Verify Paths Exist
if ~exist(datasetPath, 'dir')
    error('Training folder %s does not exist. Please check the path.', datasetPath);
end
if ~exist(testPath, 'dir')
    error('Test folder %s does not exist. Please check the path.', testPath);
end

% Image Datastore Setup (Support for multiple formats including DICOM)
imageSize = [512 512];
trainDatastore = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ... % Remove if no subfolders
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dcm'}, ...
    'ReadFcn', @(x) imresize(imread(x), imageSize)); % Custom read function for resizing

% Check if Images are Loaded
numTrainImages = numel(trainDatastore.Files);
fprintf('Found %d images in %s.\n', numTrainImages, datasetPath);
if numTrainImages == 0
    error('No images found in %s. Check folder structure and file extensions.', datasetPath);
end

% Split into Training and Validation
[trainData, valData] = splitEachLabel(trainDatastore, 0.9, 'randomize');

% Augmented Image Datastore
augTrainData = augmentedImageDatastore(imageSize, trainData, 'ColorPreprocessing', 'gray2rgb');
augValData = augmentedImageDatastore(imageSize, valData, 'ColorPreprocessing', 'gray2rgb');

% Display a Sample Image
sampleImage = readimage(trainDatastore, min(10, numTrainImages));
imshow(sampleImage);
title('Sample Image');

% Step 1: Annotate Tumors Using Image Labeler App
annotationFile = 'groundTruthAnnotations.mat';
if ~exist(annotationFile, 'file')
    disp('Annotations not found. Launching Image Labeler app...');
    imageLabeler(trainDatastore);
    disp('Please annotate tumors with bounding boxes (label: "Tumor") and export as "gTruth".');
    disp('After exporting, run: save(''groundTruthAnnotations.mat'', ''gTruth'') in the command window.');
    disp('Then rerun this script.');
    return; % Stop until annotations are ready
end

% Step 2: Load Annotations and Calculate Tumor Size
load(annotationFile, 'gTruth');
numImages = numel(gTruth.DataSource.Source);
tumorSizes = zeros(numImages, 1);
for i = 1:numImages
    roi = gTruth.LabelData{i, :};
    if ~isempty(roi) && ~isempty(roi{1})
        bbox = roi{1}; % Assume first ROI is the tumor bounding box
        tumorSizes(i) = bbox(3) * bbox(4); % Width × Height in pixels
    else
        tumorSizes(i) = NaN; % No tumor annotated
    end
end
save('tumorSizes.mat', 'tumorSizes');

% Step 3: Estimate T Stage (Simplified based on size)
tStages = cell(numImages, 1);
t1Threshold = 30; % 3 cm (adjust based on pixel-to-mm conversion)
t2Threshold = 50; % 5 cm
t3Threshold = 70; % 7 cm
for i = 1:numImages
    if isnan(tumorSizes(i))
        tStages{i} = 'Unknown';
    else
        % Approximate diameter in mm (assuming 1 pixel = 1 mm for simplicity)
        tumorSize_mm = sqrt(tumorSizes(i)); % sqrt(area) as rough diameter
        if tumorSize_mm <= t1Threshold
            tStages{i} = 'T1';
        elseif tumorSize_mm <= t2Threshold
            tStages{i} = 'T2';
        elseif tumorSize_mm <= t3Threshold
            tStages{i} = 'T3';
        else
            tStages{i} = 'T4';
        end
    end
end
save('tStages.mat', 'tStages');

% Model Architecture (Classification Only for Now)
layers = [
    imageInputLayer([512 512 3], 'Normalization', 'rescale-zero-one')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he')
    reluLayer
    maxPooling2dLayer(3, 'Stride', 3)
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(32)
    reluLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(32)
    reluLayer
    
    fullyConnectedLayer(3) % Assuming 3 classes (e.g., Benign, Malignant, Other)
    softmaxLayer
    classificationLayer
];

% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValData, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'gpu');

% Train the Model
net = trainNetwork(augTrainData, layers, options);

% Test Datastore
testDatastore = imageDatastore(testPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dcm'}, ...
    'ReadFcn', @(x) imresize(imread(x), imageSize));
augTestData = augmentedImageDatastore(imageSize, testDatastore, 'ColorPreprocessing', 'gray2rgb');

% Predictions and Actual Values
YPred = classify(net, augTestData);
YTest = testDatastore.Labels;

% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

% Display Predictions with Images, Tumor Sizes, and Stages
figure;
numDisplay = min(20, numel(testDatastore.Files));
for i = 1:numDisplay
    subplot(5, 4, i); % Adjusted to 5x4 for 20 images
    img = readimage(testDatastore, i);
    imshow(img);
    % Match test image to training annotations (simplified assumption)
    title(sprintf('True: %s\nPred: %s\nSize: %.0f px\nStage: %s', ...
        char(YTest(i)), char(YPred(i)), tumorSizes(i), tStages{i}));
end

% Save the Model
save('lung_cancer_model.mat', 'net');

disp('Script completed successfully.');