function mat_to_cwt
% Robust CWT generator for CWRU .mat files (recursive signal extraction)

inDir  = 'F:\NeuTech\CWRU';
outDir = 'F:\NeuTech\CWT';
conds  = {'ball','inner','outer','healthy'};

if ~exist(outDir,'dir'), mkdir(outDir); end

defaultFs = 12e3; % change to 48e3 if that’s your set
startIdx  = 81;   % numbering start
minLen    = 64;   % accept short signals too

for c = 1:numel(conds)
    cond = conds{c};
    matpath = resolve_mat(fullfile(inDir, cond));
    S = load(matpath);

    Fs = detect_fs_any(S, defaultFs);
    sigs = collect_signals_recursive(S, minLen);

    outSub = fullfile(outDir, cond);
    if ~exist(outSub,'dir'), mkdir(outSub); end

    fprintf('[%s] Fs=%.0f Hz, signals found: %d\n', cond, Fs, numel(sigs));
    k = 0;
    for i = 1:numel(sigs)
        x = sigs{i};
        try
            save_cwt(x, Fs, fullfile(outSub, sprintf('sample_%d.png', startIdx + k)));
            k = k + 1;
        catch ME
            warning('CWT failed on %s #%d: %s', cond, i, ME.message);
        end
    end
    fprintf('Finished %s → %d image(s) saved in %s\n', cond, k, outSub);
end
end

% ---------- helpers ----------
function p = resolve_mat(basepath)
p = [basepath '.mat'];
if ~isfile(p)
    d = dir([basepath '.*mat']);
    if ~isempty(d)
        [~,ix] = max([d.datenum]);
        p = fullfile(d(ix).folder, d(ix).name);
    else
        error('MAT file not found at %s(.mat)', basepath);
    end
end
end

function Fs = detect_fs_any(S, defaultFs)
Fs = defaultFs;
% look in common places, recursively
names = fieldnames(S);
for i = 1:numel(names)
    val = S.(names{i});
    Fs = try_detect_fs(val, Fs);
end
end

function Fs = try_detect_fs(val, Fs)
if isnumeric(val) && isscalar(val) && val > 1
    % heuristics by variable name are handled by caller
elseif istable(val) || istimetable(val)
    vnames = val.Properties.VariableNames;
    for i = 1:numel(vnames)
        col = val.(vnames{i});
        Fs = try_detect_fs(col, Fs);
    end
elseif isstruct(val)
    f = fieldnames(val);
    for i = 1:numel(f)
        name = lower(f{i});
        v = val.(f{i});
        if isnumeric(v) && isscalar(v) && v>1 ...
                && (contains(name,'fs')||contains(name,'sampl')||contains(name,'rate')||contains(name,'hz'))
            Fs = double(v); return;
        else
            Fs = try_detect_fs(v, Fs);
        end
    end
elseif iscell(val)
    for i = 1:numel(val)
        Fs = try_detect_fs(val{i}, Fs);
    end
end
end

function sigs = collect_signals_recursive(X, minLen)
% Crawl through structs/cells/tables and extract numeric vectors
sigs = {};
add = @(v) assignin('caller','sigs',[evalin('caller','sigs'); {clean_vec(v)}]); %#ok<EVAL>

if isnumeric(X)
    sigs = split_numeric(X, minLen);
elseif istable(X) || istimetable(X)
    A = table2array(X);
    sigs = split_numeric(A, minLen);
elseif isstruct(X)
    fn = fieldnames(X);
    for i = 1:numel(fn)
        v = X.(fn{i});
        % prefer CWRU names if present
        if isstruct(v)
            subfn = fieldnames(v);
            for j = 1:numel(subfn)
                name = lower(subfn{j});
                if any(strcmp(name, {'de_time','fe_time','ba_time','drive_end','fan_end','base'}))
                    add(v.(subfn{j}));
                end
            end
        end
        sigs = [sigs; collect_signals_recursive(v, minLen)]; %#ok<AGROW>
    end
elseif iscell(X)
    for i = 1:numel(X)
        sigs = [sigs; collect_signals_recursive(X{i}, minLen)]; %#ok<AGROW>
    end
end

% deduplicate by length/hash (optional)
end

function cells = split_numeric(A, minLen)
cells = {};
if ~isnumeric(A), return; end
A = double(A);
if isvector(A)
    if numel(A) >= minLen, cells = {clean_vec(A)}; end
else
    [m,n] = size(A);
    if m >= n
        for k = 1:n
            v = A(:,k);
            if numel(v) >= minLen, cells{end+1,1} = clean_vec(v); end %#ok<AGROW>
        end
    else
        for k = 1:m
            v = A(k,:);
            if numel(v) >= minLen, cells{end+1,1} = clean_vec(v); end %#ok<AGROW>
        end
    end
end
end

function v = clean_vec(v)
v = v(:);
v(~isfinite(v)) = 0;
v = v - mean(v,'omitnan');
end

function save_cwt(x, Fs, outfile)
fig = figure('Visible','off');
cwt(x,'amor',Fs);
ax = gca; ax.XLabel=[]; ax.YLabel=[]; ax.Title=[];
ax.XTickLabel={}; ax.YTickLabel={}; colorbar('off');
exportgraphics(fig, outfile, 'Resolution', 150);
close(fig);
end
