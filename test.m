% g = [1,0,1,1,0,1,1]
% generate_CS([6,7,8,11,12,13,14,15,16],4)

clear
addpath(genpath('Codes/'))

N = 128;
k = 64;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
snr_dB = 3;
pac = paccode(N,k,g,'RM');
crc_length = 8;


error=0;
for i=1:5000
u= double(rand(k,1)>0.5);

x = pac.encode(u);
sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB/20);
bpsk = 1 - 2 * x;
noise = randn(N, 1);
y = bpsk + sigma * noise;
llr = 2/sigma^2*y;
Pe=pac.get_PE(3);
d= pac.Fano_decoder(llr,Pe,1,2,39,4);
i
if(sum(sum(u~=d))>0)
    error=error+1
    
end
end

error/5000