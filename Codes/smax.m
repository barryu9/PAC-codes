function result = smax(i_start, i_cur, n)
result = 0;
for im = i_start:i_cur
    temp = ffs(im, n);
    if (temp > result)
        result = temp;
    end
end

end