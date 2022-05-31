function i_minus1 = find_sMaxPos(s_start, s_max, i_minus1, n)
ss = s_start;
while ss < s_max
    i_minus1 = i_minus1 - 2;
    if i_minus1 > 0
        ss = ffs(i_minus1, n);
    else
        ss = n;
    end
end
end
