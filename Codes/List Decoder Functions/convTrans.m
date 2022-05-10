function u = convTrans(v,c)
    u=zeros(1,length(v));
    curr_state = zeros(1,length(c)-1);
    for i = 1:length(v)
        [u(i),curr_state]=conv1bTrans(v(i),curr_state,c);
    end
end
