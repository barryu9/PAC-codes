clear
addpath(genpath('Codes/'))

N = 8;
k = 4;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
snr_dB = 1;
pac = paccode(N,k,g,'GA',2);




u= double(rand(k,1)>0.5);
x = pac.encode(u);
sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB/20);
bpsk = 1 - 2 * x;
noise = randn(N, 1);
y = bpsk + sigma * noise;
llr = 2/sigma^2*y;
Pe=[0.99,0.99,0.99,0.01,0.99,0.01,0.01,0.01];
d= pac.Fano_decoder(llr,Pe,1,2,4,4)
u