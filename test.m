N=8;
n=ceil(log2(N));
N=2^n;
k=4;
R=k/N;
d=[0,1,1,0];
c=[1,0,1];
m=length(c);
snr_dB=20;
%% RM score
Channel_indices=(0:N-1)';
bitStr=dec2bin(Channel_indices);
bit=abs(bitStr)-48;
RM_score=sum(bit,2);
[RM_score_sorted, sorted_indices]=sort(RM_score,'ascend');
info_indices=sorted_indices(end-k+1:end);
%% Rate Profile
v=zeros(1,N);
v(info_indices)=d;
%% convolutional encoder
c_zp=[c,zeros(1,N-m)];
T=toeplitz(c_zp);
u=v*T;
%% Polar Encoding
P=get_P(N);
x=mod(u*P,2);
bpsk=1-2*x;
%% Channel
sigma = 1/sqrt(2 * R) * 10^(-snr_dB/20);
noise = randn(1,N);
y=bpsk+sigma*noise;
%% List decoding
llr=2*y/sigma^2;


