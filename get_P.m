function P = get_P(N)
F = [1, 0 ; 1, 1];
P = zeros(N, N);
P(1 : 2, 1 : 2) = F;
for i = 2 : log2(N)
    P(1 : 2^i, 1 : 2^i) = kron(P(1 : 2^(i - 1), 1 : 2^(i - 1)), F);
end
