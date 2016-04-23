% Train and test the backgroud noise of a device
% predict need two files: map.mat and model.mat.
for runningmode=1:2 ;  %  1. training , 2. test
freq = 44100; % sampling rate. Feature和采样率无关的，主要又取样长度决定
samlen =  4096; %16384;%8192; %4096; %16384;  % length of samole.  = 2^14
NFFT = 2^nextpow2(samlen); % Next power of 2 from length of y
dimneed = NFFT/2+1;
lev = 1;  % 滤波降噪的Level
homepath = pwd;
% get all files under train/test directory:
if runningmode == 1
    datapath = sprintf('%s\\%s',homepath,'traindata');
else
    datapath = sprintf('%s\\%s',homepath,'testdata');
end
wavfiles = dir(fullfile(datapath,'*.wav'));

%% prepare the training/test set.
fnum = size(wavfiles,1);
y = zeros(fnum,1);
if runningmode == 1
    map = java.util.Hashtable; 
    ycnt = 0;
else
    load map.mat;
    ycnt= size(map);
end
x = zeros(dimneed,fnum);
for ids=1:fnum
    fname = wavfiles(ids).name;
    [pathstr,name,ext] = fileparts(fname);
    hypi = strfind(name,'-');
    label = name(1:hypi-1);
    yt = map.get(label);
    if isempty(yt)==1
        ycnt=ycnt+1;
        map.put(label,ycnt);
        yt = ycnt;
    end
    y(ids)=yt;
    infile = sprintf('%s\\%s',datapath, fname);
    [wy, Fs, nbits, readinfo] = wavread(infile);
    % if need to cut the 功率较大部分？
    % calcu the x:
    wavlen = size(wy,1);
    %f = Fs/2*linspace(0,1,NFFT/2+1);
    yd = wden(wy,'minimaxi','s','sln',lev+16,'sym8');
    %yd = wden(wy,'heursure','s','mln',lev+16,'sym8');
    %yd = wden(wy,'minimaxi','s','mln',lev+16,'sym8');
    %yd = wden(wy,'minimaxi','s','mln',lev+16,'sym8');
    %yd1 = wden(yd,'minimaxi','s','sln',lev+4,'db2');
    %yd = wden(yn1,'minimaxi','s','sln',lev+3,'sym8');
    yn=wy-yd;
    %yn=wy;
    %yd1 = wden(yn,'minimaxi','s','mln',lev+8,'db2');
    %yn = yn - yd1;
    %identifier= 'MATLAB:oldPfileVersion';
    %warning('off',identifier)
    %yn = GA_SE_New(yn1,Fs,nbits);
    %yn = deharmornic(yn1);
    %yfft = fft(wy,NFFT);
    %vy = abs(yfft(1:dimneed));
    %vs = log(vy+1)+1;
    fftyn = fft(yn,NFFT);
    %fftyd = fft(yd,NFFT);
    %fftwy = fft(wy,NFFT);
   
    vf = abs(fftyn(1:dimneed));
    %vf = real(fftyn(1:dimneed));
    %vf = tanh(vf);
    %vf = logsig(vf);
    %vf = abs(fftwy-fftyd);
    %vf = vf(1:dimneed);
    vf = log(vf+1);
    %vf = vf/norm(vf);
    %vf = (vf-min(vf)) / (max(vf)-min(vf));
    %vf = (vf-mean(vf)) ./ (max(vf)-min(vf));
    x(:,ids) = vf;
    fprintf('gen x for %s \n',fname);
end
if runningmode == 1 
    save map.mat map;
end

%[nx, mu, sigma] = zscore(x,0,2);
%[x, mu, sigma] = zscore(x);
% save devicefeature.mat x mu sigma;

% if runningmode ==1 
%     save devicefeature.mat x mu sigma;
% else
%     xtemp = x;
%     load devicefeature.mat;
%     x = normalize(xtemp, mu, sigma);
% end    
if runningmode == 1 
    save train.mat x y;
else
    save test.mat x y;
end
%% do the train / test
inputSize = dimneed;
numClasses = 9;
lambda = 1e-3;         % weight decay parameter       1e-3
%%======================================================================
%[stackedAEOptTheta,netconfig] = stackedAETrain(inputSize,numClasses ,hiddenSizeL1,hiddenSizeL2,sparsityParam,lambda,beta,x',y,maxIterNum);
%[pred, allprob] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL1, hiddenSizeL2,numClasses, netconfig, x');
if runningmode==1
    options.display = 'on';
    softmaxModel = softmaxTrain(inputSize, numClasses, lambda, x, y, options);
    save model.mat softmaxModel;
else
    load model.mat;
end
[pred] = softmaxPredict(softmaxModel, x);

fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == y(:)));

end
