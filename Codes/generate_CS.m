function CS_logical = generate_CS(rate_profiling,n)
info_bits = zeros(1,2^n);
info_bits(rate_profiling)=1;
info_bits=logical(info_bits);

B=-1*ones(n+1,2^n);
cnt=1;
for k=1:2^n
    B(n+1,k)=0;
    if(info_bits(k)==0)
        B(n+1,k)=1;
    end
end

for i=n:-1:1
    for j=1:2^(i-1)
        B(i,j)=B(i+1,2*j-1)+B(i+1,2*j);
    end
end

for i=1:n+1
    for j=1:2^(i-1)
        x1=j;
        x2=j;
        if(B(i,j)==0)
            for k=1:(n+1)-i
                x1=2*x1-1;
                x2=2*x2;
                for p=x1:x2
                    B(i+k,p)=-1;
                end
            end
            CS(cnt)=x1;
            cnt=cnt+1;
        end
    end
end

CS_logical = zeros(1,2^n);
CS_logical(CS) = 1;

end

