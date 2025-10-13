[y,Fs]=audioread('recording.WAV');
N = length(y);
%计算音频时域绘图数据
t = (0:N-1)/Fs;
%计算频域绘图数据
Y = fft(y(:,1));
f = (0:N-1)*Fs/N;
subplot(221);plot(t,y(:,1));%时域波形
subplot(222);plot(f,abs(Y));%频域波形

% 给原始语音信号加噪声
% 假设噪声强度为原始信号幅度的5%
noiseLevel = 0.05;
noise = noiseLevel * randn(size(y));
noisy_y = y + noise;

% 计算加噪声后的语音信号的时域绘图数据
noisy_t = (0:N-1)/Fs;

% 计算加噪声后的语音信号的频域绘图数据
noisy_Y = fft(noisy_y(:,1));
noisy_f = (0:N-1)*Fs/N;

% 绘制加噪声后的语音信号的时域波形和频谱图
subplot(223); plot(noisy_t, noisy_y(:,1)); % 加噪声后的时域波形
subplot(224); plot(noisy_f, abs(noisy_Y)); % 加噪声后的频域波形


% 定义带通频率范围
Fc_low = 150;   % 低截止频率（Hz）
Fc_high = 300; % 高截止频率（Hz）

% 创建归一化频率向量
norm_freq = (0:N-1)/N;

% 根据归一化频率设置带通信道增益
channelGain = zeros(N, 1);
channelGain(Fc_low/(Fs/2) <= norm_freq & norm_freq <= Fc_high/(Fs/2)) = 1;

% 应用理想带通信道进行幅度调制
modulated_Y = noisy_Y .* channelGain;

% 计算调制后的信号的逆FFT以获取时域信号
modulated_y = ifft(modulated_Y);

% 绘制调制后的语音信号的时域波形
subplot(231); plot(noisy_t, real(modulated_y)); % 调制后的时域波形
title('调制后的时域波形');
% 绘制调制后的频谱图
subplot(232); plot(f(1:N/2+1), abs(modulated_Y(1:N/2+1))); % 调制后的频域波形
title('调制后的频域波形');

% 假设接收端没有额外噪声
received_Y = modulated_Y;

% 对接收信号进行解调（解调就是IFFT）
demodulated_Y = ifft(received_Y);  % 直接对接收信号进行IFFT

% 由于IFFT的结果通常是复数，我们需要取实部进行滤波
demodulated_y = real(demodulated_Y);

% 绘制解调后的语音信号的时域波形
subplot(233); plot(noisy_t, real(demodulated_Y)); % 解调后的时域波形
title('解调后的时域波形');

% 绘制解调后的频谱图
subplot(234); plot(noisy_f, abs(demodulated_Y)); % 解调后的频域波形
title('解调后的频域波形');

% 设计低通滤波器参数
filterOrder = 50;  % 滤波器的阶数，可以根据需要调整
cutoffFrequency = 200;  % 截止频率，可以根据带通滤波器的带宽调整

% 使用fir1函数设计FIR低通滤波器
filterCoeffs = fir1(filterOrder, cutoffFrequency / (Fs / 2), 'low');

% 应用低通滤波器
% 由于filter函数期望的输入是行向量，因此需要调整维度
filtered_y = filtfilt(filterCoeffs, 1,demodulated_y);

% 播放滤波后的语音信号
disp('正在播放滤波后的语音信号，请听...');
sound(filtered_y, Fs);

% 暂停一秒以确保播放完成
pause(length(filtered_y) / Fs + 1);

% 重新播放原始语音信号进行对比
disp('现在播放原始语音信号...');
sound(y, Fs);
