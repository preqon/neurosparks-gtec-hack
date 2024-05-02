// this matlab script applies a bandpass filter between 8 and 30 Hz 
eegData=load('P1_pre_training.mat') // P1_pre_training.mat is substituted with other file names
fs = 256;
[b, a] = butter(4, [8 30] / (fs / 2), 'bandpass'); % Butterworth filter
filteredData = filter(b, a, y')';

// to all participant/treatments
