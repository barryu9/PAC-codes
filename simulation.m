clear
addpath(genpath('Codes/'))

N = 512;
k = 256;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
snr_dB = 0:0.25:3;
pac = paccode(N,k,g,'GA',2);
n_iter=1e5;
frame_errors_count=zeros(1,length(snr_dB));
bit_errors_count=zeros(1,length(snr_dB));
FER=zeros(1,length(snr_dB));
BER=zeros(1,length(snr_dB));
L=256;

parfor i=1:length(snr_dB)
    for ii = 1:n_iter
        u= double(rand(k,1)>0.5);
        x = pac.encode(u);
        sigma = 1/sqrt(2 * pac.R) * 10^(-snr_dB(i)/20);
        bpsk = 1 - 2 * x;
        noise = randn(N, 1);
        y = bpsk + sigma * noise;
        llr = 2/sigma^2*y;
        d= pac.SCL_decoder(llr,L);
        errs=sum(sum(u~=d));
        if(errs>0)
            frame_errors_count(i)=frame_errors_count(i)+1;
            bit_errors_count(i)=bit_errors_count(i)+errs;
        end
        if(mod(ii, 100)==0)
            display_info(N,k,snr_dB(i),ii,n_iter,L,frame_errors_count(i),bit_errors_count(i));
        end
    end

end

FER=frame_errors_count/n_iter;
BER=bit_errors_count/(n_iter*k);
save(['results\PAC_',datestr(datetime('now'),'yyyy-mm-dd-HH-MM'),'.mat'])

figure;
semilogy(snr_dB,FER,'-o','LineWidth',1);
grid on;

function display_info(N,k,snr_dB,iter_count,n_iter,L,frame_errors_count,bit_errors_count)
disp(' ');
disp(['Sim iteration running = ' num2str(iter_count) '/' num2str(n_iter)]);
disp(['N = ' num2str(N) ' K = ' num2str(k)]);
disp(['List size = ' num2str(L)]);
disp('SNR       BLER         BER');
disp(num2str([snr_dB  frame_errors_count./iter_count bit_errors_count./(iter_count*k)]));
disp(' ')
end
