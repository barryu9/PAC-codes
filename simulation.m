clear
addpath(genpath('Codes/'))
tic;
N = 128;
k = 64;
g = [1,0,1,1,0,1,1];%c=[c_0,c_1,...,c_m]
% snr_dB = 0:0.25:3;
snr_dB = 3;

Rate_Profiling_method = 'RM-Polar';
dsnr = 3.5;
crc_length = 0;

pac = paccode(N,k,g,crc_length,Rate_Profiling_method,dsnr);
% n_iter=[100,100,100,150,200,250,300,400,500,5000,5000,50000,50000];
n_iter=[50000];

frame_errors_count=zeros(1,length(snr_dB));
operation_count_C_c=zeros(1,length(snr_dB));
operation_count_P_c=zeros(1,length(snr_dB));

bit_errors_count=zeros(1,length(snr_dB));
FER=zeros(1,length(snr_dB));
BER=zeros(1,length(snr_dB));
L=32;
pe=zeros(1,N);
delta=1;
global operation_count_C;
global operation_count_P;
for i=1:length(snr_dB)
    for ii = 1:n_iter(i)
        operation_count_C=0;
        operation_count_P=0;

        u= double(rand(k,1)>0.5);
        x = pac.encode(u);
        sigma = 1/sqrt(2 * pac.rate) * 10^(-snr_dB(i)/20);
        bpsk = 1 - 2 * x;
        noise = randn(N, 1);
        y = bpsk + sigma * noise;
        llr = 2/sigma^2*y;
        %         d = pac.Fano_decoder(llr,pe,delta);
        %         d = pac.SCL_decoder(llr,256);
        d = pac.Viterbi_decoder(llr,4);

        operation_count_C_c(i)=operation_count_C_c(i)+operation_count_C;
        operation_count_P_c(i)=operation_count_P_c(i)+operation_count_P;

        errs=sum(sum(u~=d));
        if(errs>0)
            frame_errors_count(i)=frame_errors_count(i)+1;
            bit_errors_count(i)=bit_errors_count(i)+errs;
        end
        if(mod(ii, 1)==0)
            display_info(N,k,snr_dB(i),ii,n_iter(i),L,frame_errors_count(i),bit_errors_count(i),operation_count_C_c(i),operation_count_P_c(i));
        end
    end

end

avg_operation_count_C=operation_count_C./n_iter;
avg_operation_count_P=operation_count_P./n_iter;

FER=frame_errors_count./n_iter;
BER=bit_errors_count./(n_iter.*k);
save(['results\PAC_',datestr(datetime('now'),'yyyy-mm-dd-HH-MM'),'.mat'])

figure;
semilogy(snr_dB,FER,'-o','LineWidth',1);
grid on;
toc;

function display_info(N,k,snr_dB,iter_count,n_iter,L,frame_errors_count,bit_errors_count,operation_count_C,operation_count_P)
disp(' ');
disp(['Sim iteration running = ' num2str(iter_count) '/' num2str(n_iter)]);
disp(['N = ' num2str(N) ' K = ' num2str(k)]);
% disp(['List size = ' num2str(L)]);
disp('SNR       BLER         BER         avg_C_count         avg_P_count         total_count');
disp(num2str([snr_dB  frame_errors_count./iter_count bit_errors_count./(iter_count*k) operation_count_C./iter_count operation_count_P./iter_count (operation_count_C+operation_count_P)./iter_count]));
disp(' ')
end


