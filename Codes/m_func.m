function m = m_func(lambda_0,u)
m = -log2(1+exp(-(1-2*u)*lambda_0));
% if(u==0.5*(1-sign(lambda_0)))
%     m = abs(lambda_0);
% else
%     m = 0;
% end
end