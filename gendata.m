% read a bunch of wave file , generate n traning samples each 
% 随机CUT，还是只是选择功率较小的地方？
% ausource为WAV源文件所在目录。
rand('seed',555); 
maklen = 4096 ; %2048; %16384;%8192;
maknum = 1000;  
manum=10;  % 移动平均的长度
homepath = pwd;
% get all files under ausource directory:
wavfiles = dir(fullfile(homepath,'ausource\\train\\*.wav'));
fnum = size(wavfiles,1);
for ids=1:fnum
    fname = wavfiles(ids).name;
    [pathstr,name,ext] = fileparts(fname);
    devicename = name;
    %hypi = strfind(name,'-');
    %if hypi>0
    %    devicename = name(1:hypi-1);
    %end
    infile = sprintf('%s\\%s\\%s',homepath, 'ausource\\train',fname);
    [y, Fs, nbits, readinfo] = wavread(infile);
    % if need to cut the 功率较大部分？
     wavlen = size(y,1);
     irange = wavlen-maklen;
%     mask = ones(manum,1)/manum;
%     my = conv(abs(y),mask,'same');
%     meany = mean(abs(y));
%     stdy = std(abs(y));
%     thresh = meany*1.0;
    for i=1:maknum
        st1 = unidrnd(irange);
%         while my(st1+maklen)>thresh   % 结束的地方是低电平
%             st1 = unidrnd(irange);
%         end
%         st2 = unidrnd(irange);
%         while my(st2)>thresh   % 开始的地方是低电平
%             st2 = unidrnd(irange);
%         end
        x= y(st1:st1+maklen-1);
        filename = sprintf('%s\\%s\\%s%s%04d%s',homepath, 'traindata',devicename,'-',i,'.wav');
        wavwrite(x,Fs,nbits,filename)
        fprintf('gen %d for %s, len: %d  .  %d \n',i,fname, wavlen,st1);
    end
end

%% generate test

maknum = 100;  
wavfiles = dir(fullfile(homepath,'ausource\\test\\*.wav'));
fnum = size(wavfiles,1);
for ids=1:fnum
    fname = wavfiles(ids).name;
    [pathstr,name,ext] = fileparts(fname);
    devicename = name;
    %hypi = strfind(name,'-');
    %if hypi>0
    %    devicename = name(1:hypi-1);
    %end
    infile = sprintf('%s\\%s\\%s',homepath, 'ausource\\test',fname);
    [y, Fs, nbits, readinfo] = wavread(infile);
    % if need to cut the 功率较大部分？
     wavlen = size(y,1);
     irange = wavlen-maklen;
%     mask = ones(manum,1)/manum;
%     my = conv(abs(y),mask,'same');
%     meany = mean(abs(y));
%     stdy = std(abs(y));
%     thresh = meany*1.0;
    for i=1:maknum
        st1 = unidrnd(irange);
%         while my(st1+maklen)>thresh   % 结束的地方是低电平
%             st1 = unidrnd(irange);
%         end
%         st2 = unidrnd(irange);
%         while my(st2)>thresh   % 开始的地方是低电平
%             st2 = unidrnd(irange);
%         end
        x= y(st1:st1+maklen-1);
        filename = sprintf('%s\\%s\\%s%s%04d%s',homepath, 'testdata',devicename,'-',i,'.wav');
        wavwrite(x,Fs,nbits,filename)
        fprintf('gen %d for %s, len: %d  . %d \n',i,fname, wavlen,st1);
    end
end
