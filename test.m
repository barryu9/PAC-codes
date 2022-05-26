% g = [1,0,1,1,0,1,1]
% generate_CS([6,7,8,11,12,13,14,15,16],4)

clear
addpath(genpath('Codes/'))

N = 8;
k = 4;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
snr_dB = 5;
pac = paccode(N,k,g,0,'GA',3);

Pe=pac.get_PE_GA(3);
sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB/20);

error=0;
for i=1:500
    u= double(rand(k,1)>0.5);
    x = pac.encode(u);
    bpsk = 1 - 2 * x;
    noise = randn(N, 1);
    y = bpsk + sigma * noise;
    llr = 2/sigma^2*y;
    [d]= pac.My_Fano_decoder(llr,Pe,1);

    if(sum(sum(u~=d))>0)
        error=error+1;
    end
end

error/i