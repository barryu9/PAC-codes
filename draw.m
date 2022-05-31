figure;
semilogy(snr_dB, FER, '-o', 'LineWidth', 1);
% semilogy(SNRdBNA,Pev,'-','LineWidth',1);
axis([1, 3, -inf, 1e-5])
% legend('SCL-8, RM','SCL-8, GA','C8ASCL-8, GA','SCL-256, GA','Dispersion Bound')
% legend('SCL-32, GA','SCL-32, RM','SCL-256, RM','C8ASCL-256, RM','Viterbi, L=4, m=5, GA','Dispersion Bound')

title('PAC Codes (128,64)')
xlabel('SNR')
ylabel('FER')
hold on
grid on;