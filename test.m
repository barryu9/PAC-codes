% g = [1,0,1,1,0,1,1]
% generate_CS([6,7,8,11,12,13,14,15,16],4)

clear
addpath(genpath('Codes/'))

N = 128;
k = 64;
g = [1, 0, 1, 1, 0, 1, 1]; %c=[c_0,c_1,...,c_m]
snr_dB = 1.5;
pac = paccode(N, k, g, 0, 'RM-Polar', 3);

Pe = pac.get_PE_GA(4);
sigma = 1 / sqrt(2*pac.R) * 10^(-snr_dB / 20);

%
% u=[0;1;1;0];
% llr = [-6.24841359764031	1.32876011828409	-1.42233129484890	4.05711149046604	-3.63127705148068	2.81019082828330	1.54540323409445	-3.36647459351069]';
% [d]= pac.My_Fano_decoder(llr,Pe,1);

error = 0;
for i = 1:5000
    u = double(rand(k, 1) > 0.5);
    x = pac.encode(u);
    bpsk = 1 - 2 * x;
    noise = randn(N, 1);
    y = bpsk + sigma * noise;
    llr = 2 / sigma^2 * y;
    [d] = pac.My_Fano_decoder(llr, Pe, 1);
    i
    if (sum(sum(u ~= d)) > 0)
        error = error + 1;
        error
    end
end

error / i