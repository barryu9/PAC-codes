N = 8;
k = 4;
g = [1,0,1];%c=[c_0,c_1,...,c_m]
snr_dB = 20;
d = [1,1,1,0];
pac = paccode(N,k,g,'RM');
x = pac.encode(d);
%% Channel
sigma = 1/sqrt(2 * R) * 10^(-snr_dB/20);
noise = randn(1,N);
y = bpsk + sigma * noise;
%% List decoding
llr = 2 * y / sigma^2;
