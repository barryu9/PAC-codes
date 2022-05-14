function result = smax(i_start,i_cur)
result = 0;
for im = i_start:i_cur
    temp = ffs(im);
    if(temp>result)
        result = temp;
    end
end

end