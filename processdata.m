clear all 
clc
% 
% % Define the input and output directories
% inputDir = 'C:\Users\harry\OneDrive - Imperial College London\My Modules\EEG_Comp\data\data_neuralink';
% outputDir = 'C:\Users\harry\OneDrive - Imperial College London\My Modules\EEG_Comp\data\neuralink_matdata';
% 
% % Create the output directory if it doesn't exist
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end
% 
% % Get a list of all .wav files in the input directory
% wavFiles = dir(fullfile(inputDir, '*.wav'));
% 
% % Process each .wav file
% for k = 1:length(wavFiles)
%     % Get the full path of the .wav file
%     wavFilePath = fullfile(inputDir, wavFiles(k).name);
% 
%     % Read the .wav file
%     [audioData, sampleRate] = audioread(wavFilePath);
% 
%     % Generate time values
%     numSamples = length(audioData);
%     timeValues = (0:numSamples-1) / sampleRate; % Time vector in seconds
% 
%     % Define the output .mat file path
%     [~, fileName, ~] = fileparts(wavFilePath);
%     matFilePath = fullfile(outputDir, [fileName, '.mat']);
% 
%     % Save the raw data and time values to a .mat file
%     save(matFilePath, 'audioData', 'timeValues', 'sampleRate');
% 
%     % Display progress
%     fprintf('Processed and saved: %s\n', matFilePath);
% end

% disp('All files processed and saved successfully.');

% Define the directory containing the .mat files
dataDir = 'C:\Users\harry\OneDrive - Imperial College London\My Modules\EEG_Comp\data\BCI_IV_dataset4\BCICIV_4_mat';

% List of .mat files to process
matFiles = {'sub1_comp.mat', 'sub2_comp.mat', 'sub3_comp.mat'};

% Sampling frequency for BCI data
samplingFrequency = 1000; % in Hz

% Number of channels for each subject
numChannels = [62, 48, 64];

% Process each file
for k = 1:length(matFiles)
    % Load the .mat file
    matFilePath = fullfile(dataDir, matFiles{k});
    data = load(matFilePath);
    
    % Delete the train_dg variable if it exists
    if isfield(data, 'train_dg')
        data = rmfield(data, 'train_dg');
    end
    
    % Concatenate test_data onto the bottom of train_data
    if isfield(data, 'train_data') && isfield(data, 'test_data')
        data.train_data = [data.train_data; data.test_data];
    end
    
    % Generate time values for the concatenated train_data
    numSamples = size(data.train_data, 1);
    timeValues = (0:numSamples-1) / samplingFrequency; % Time vector in seconds
    
    % Add the time values to the data struct
    data.timeValues = timeValues;
    
    % Separate each channel into its own variable
    for ch = 1:numChannels(k)
        channelName = sprintf('channel_%02d', ch);
        data.(channelName) = data.train_data(:, ch);
    end
    
    % Remove the original train_data variable
    data = rmfield(data, 'train_data');
    
    % Save the modified data back into the .mat file
    save(matFilePath, '-struct', 'data');
    
    % Display progress
    fprintf('Processed and saved: %s\n', matFilePath);  
end

disp('All files processed and saved successfully.');
