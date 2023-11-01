figure;
semilogy(snr_dB, complexity(1,:), '-o', 'LineWidth', 1);

% semilogy(SNRdBNA,Pev,'-','LineWidth',1);
% axis([0.5, 3, 1e-5, 1])
% legend('Fano','SCL-32, GA','SCL-32, RM','SCL-256, RM','C8ASCL-256, RM','Viterbi, L=4, m=6, RM','Dispersion Bound')

title('PAC Codes (128,64)')
xlabel('SNR')
ylabel('Average Computation Times')
hold on
grid on;

semilogy(snr_dB, complexity(2,:), '-o', 'LineWidth', 1);
semilogy(snr_dB, complexity(3,:), '-o', 'LineWidth', 1);
semilogy(snr_dB, complexity(3,:), '-v', 'LineWidth', 1);
legend('Fano','SCL-32','SCL-256','Viterbi L=4,m=6')
