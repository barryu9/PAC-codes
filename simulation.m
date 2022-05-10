N = 512;
k = 256;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
snr_dB = 100;
d = double(rand(k,1)>0.5);
pac = paccode(N,k,g,'RM');
x = pac.encode(d);
sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB/20);
bpsk = 1 - 2 * x;
noise = randn(N, 1);
y = bpsk + sigma * noise;
llr = 2/sigma^2*y;
d_d= pac.SCL_decoder(llr,1);
