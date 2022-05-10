N = 8;
k = 4;
g = [1,0,0];%c=[c_0,c_1,...,c_m]
snr_dB = 100;
d = [1,0,0,1];
pac = paccode(N,k,g,'RM');
x = pac.encode(d);
sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB/20);
bpsk = 1 - 2 * x;
noise = randn(1, N);
y = bpsk + sigma * noise;
llr = 2/sigma^2*y;
d_d= pac.SCL_decoder(llr,4);