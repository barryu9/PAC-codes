function location = ffs(i)
bin = abs(dec2bin(i))-48;
location = length(bin)  - 1;
for j = length(bin):-1:1
    if(bin(j)==1)
        location = j - 1;
        break;
    end
end
end