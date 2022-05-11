function ZWi = get_BEC_ZWi(N, ZW)
ZWi = zeros(N, 1);
ZWi(1) = ZW;
m = 1;
while(m <= N/2)
    for k = 1 : m
        Z_tmp = ZWi(k);
        ZWi(k) = 2 * Z_tmp - Z_tmp^2;%use upper bound
        ZWi(k + m) = Z_tmp^2;
    end
    m = m * 2;
end
ZWi = bitrevorder(ZWi);%perform bit-reversal order.