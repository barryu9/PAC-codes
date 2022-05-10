function PM = calc_PM(PM,llr,u)
    if(u~=0.5*(1-sign(llr)))
        PM=PM+abs(llr);
    end

end

