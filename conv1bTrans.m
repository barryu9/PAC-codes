function [u,next_state] = conv1bTrans(v,curr_state,c)
u=mod(v*c(1),2);
for j=2:length(c)
    if(c(j)==1)
        u=mod(u+curr_state(j-1),2);
    end
end
next_state=[v,curr_state(1:end-1)];
end