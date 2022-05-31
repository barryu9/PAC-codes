function location = ffs(i, n)
if i == 0
    location = n - 1;
    return
end
bin = abs(dec2bin(i, n)) - 48;
location = 0;
for j = length(bin):-1:1
    if (bin(j) == 1)
        return;
    end
    location = location + 1;
end
end