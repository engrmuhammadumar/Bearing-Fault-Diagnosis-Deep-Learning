file_path = 'D:\Concrete RUL\STREAM20191203-105108-756.wfs';  % your .wfs file
output_dir = 'F:\concrete test 1\';                % where to save chunks

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for i = 0:1036  % 1037 seconds
    try
        fprintf('Processing chunk %d/%d...\n', i+1, 1037);
        [signals, t, fs, nch] = wfsread_exp(file_path, i, i+1);
        
        % Save chunk with padded name (e.g., chunk_0001.mat)
        save(fullfile(output_dir, sprintf('chunk_%04d.mat', i)), ...
             'signals', 't', 'fs', 'nch', '-v7.3');
             
    catch ME
        warning('Error at chunk %d: %s', i, ME.message);
    end
end

disp('✅ All chunks saved successfully!');
