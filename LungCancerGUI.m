function LungCancerDetectionGUI()
    % Create Main GUI Figure (Wider for Better Layout)
    fig = figure('Name', 'Lung Cancer Detection', 'Position', [100, 100, 1200, 650]);
    movegui(fig, 'center');

    % UI Panel for Menu (Smaller Width & Higher Positioning of Buttons)
    menuPanel = uipanel('Title', 'Menu', 'FontSize', 12, 'FontWeight', 'bold', ...
        'Position', [0.005, 0.05, 0.14, 0.9]); % Reduced width, shifted slightly

    % **Shifted Menu Buttons UP**
    uicontrol(menuPanel, 'Style', 'pushbutton', 'String', 'Browse Input', 'FontSize', 12, ...
        'Position', [10, 400, 130, 40], 'Callback', @browseImage); 
    uicontrol(menuPanel, 'Style', 'pushbutton', 'String', 'Processing', 'FontSize', 12, ...
        'Position', [10, 340, 130, 40], 'Callback', @processImage);
    uicontrol(menuPanel, 'Style', 'pushbutton', 'String', 'Detection', 'FontSize', 12, ...
        'Position', [10, 280, 130, 40], 'Callback', @classifyImage);
    uicontrol(menuPanel, 'Style', 'pushbutton', 'String', 'Reset', 'FontSize', 12, ...
        'Position', [10, 220, 130, 40], 'Callback', @resetGUI);
    uicontrol(menuPanel, 'Style', 'pushbutton', 'String', 'Exit', 'FontSize', 12, ...
        'Position', [10, 160, 130, 40], 'Callback', @(~,~) close(fig));

    % **Axes for Displaying Images (Further Shifted Right to Create Space)**
    ax1 = subplot(2, 3, 1, 'Parent', fig); title(ax1, 'Input Image');
    ax2 = subplot(2, 3, 2, 'Parent', fig); title(ax2, 'Filtered Image');
    ax3 = subplot(2, 3, 3, 'Parent', fig); title(ax3, 'Edge Detection');
    ax4 = subplot(2, 3, 4, 'Parent', fig); title(ax4, 'Lung Extraction');
    ax5 = subplot(2, 3, 5, 'Parent', fig); title(ax5, 'Segmentation');
    ax6 = subplot(2, 3, 6, 'Parent', fig); title(ax6, 'Filtered Image'); % Updated Label

    % **Shifted Detection Result Panel Below for Clear View**
    resultPanel = uipanel('Title', 'Detection Result', 'FontSize', 12, 'FontWeight', 'bold', ...
        'Position', [0.18, 0.01, 0.8, 0.1]);  
    resultText = uicontrol(resultPanel, 'Style', 'text', 'String', 'N/A', ...
        'FontSize', 14, 'ForegroundColor', 'blue', ...
        'Position', [20, 10, 800, 30], 'HorizontalAlignment', 'center');

    % Load Trained Model
    model = load('lung_cancer_model.mat');
    model = model.net;

    % Initialize Image Variable
    img = [];

    % Function to Browse and Load Image
    function browseImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select Lung Scan');
        if isequal(file, 0), return; end
        img = imread(fullfile(path, file));
        imshow(img, 'Parent', ax1);
        title(ax1, 'Input Image');
    end

    % Function for Image Processing
    function processImage(~, ~)
        if isempty(img), errordlg('Load an image first!', 'Error'); return; end
        grayImg = rgb2gray(img);
        filteredImg = imgaussfilt(grayImg, 2);
        edgeImg = edge(filteredImg, 'canny'); % Edge Detection
        segmentedImg = imbinarize(filteredImg, 'adaptive'); % Segmentation

        % **Lung Extraction Using Edge Detection**
        lungExtracted = edge(grayImg, 'Canny');

        % **Extract Only Tumor (New Output Image)**
        threshold = graythresh(filteredImg);
        binaryImg = imbinarize(filteredImg, threshold);
        binaryImg = imopen(binaryImg, strel('disk', 5)); % Remove small objects
        binaryImg = imclose(binaryImg, strel('disk', 7)); % Close gaps in tumor

        % Get Largest Connected Component (Assuming it's the Tumor)
        cc = bwconncomp(binaryImg);
        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, idx] = max(numPixels);
        tumorMask = false(size(binaryImg));
        tumorMask(cc.PixelIdxList{idx}) = true;

        % Extract Tumor from Original Image
        extractedTumor = img;
        extractedTumor(repmat(~tumorMask, [1, 1, 3])) = 0; % Black Background

        % Display Processed Images
        imshow(filteredImg, 'Parent', ax2); title(ax2, 'Filtered Image');
        imshow(edgeImg, 'Parent', ax3); title(ax3, 'Edge Detection');
        imshow(lungExtracted, 'Parent', ax4); title(ax4, 'Lung Extraction');
        imshow(segmentedImg, 'Parent', ax5); title(ax5, 'Segmentation');
        imshow(extractedTumor, 'Parent', ax6); title(ax6, 'Filtered Image'); % Final Tumor Output
    end

    % Function for Lung Cancer Classification
    function classifyImage(~, ~)
        if isempty(img), errordlg('Load an image first!', 'Error'); return; end
        imgResized = imresize(img, [512, 512]);
        label = classify(model, imgResized);
        set(resultText, 'String', char(label));
    end

    % Function to Reset GUI
    function resetGUI(~, ~)
        cla(ax1); cla(ax2); cla(ax3); cla(ax4); cla(ax5); cla(ax6);
        set(resultText, 'String', 'N/A');
    end
end