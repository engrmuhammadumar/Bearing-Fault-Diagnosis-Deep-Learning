% Assuming fs is the sampling frequency and signal is your signal matrix
% Load the workspace variables if needed
% load('your_file.mat'); 

% Time vector (assuming continuous sampling)
t = (0:size(signal, 2) - 1) / fs;

% Plot and save Channel 1 separately
figure;
plot(t, signal(1,:), 'Color', [1, 0.5, 0]);  % RGB for orange
xticks([]); % Remove x-axis calibration
yticks([]); % Remove y-axis calibration
% set(gcf, 'Position',  [100, 100, 800, 600]); % Optional: set figure size
saveas(gcf, 'E:\Collaboration Work\With Farooq\Bearings\vibration_signal_display_channel_1.png');

% Plot and save Channel 2 separately
figure;
plot(t, signal(2,:), 'Color', [1, 0.5, 0]);  % RGB for orange
xticks([]); % Remove x-axis calibration
yticks([]); % Remove y-axis calibration
title('Channel 2 Vibration Signal');
xlabel('Time (s)');
ylabel('Amplitude');
set(gcf, 'Position',  [100, 100, 800, 600]); % Optional: set figure size
saveas(gcf, 'E:\Collaboration Work\With Farooq\Bearings\vibration_signal_display_channel_2.png');
