classdef paccode
    %PACCODE 定义一个paccode类
    properties
        N %码长
        k %信息长度
        g %卷积生成序列
        n %n=log2(N)
        R %码率
        rate_profiling %码率分布
        conv_depth %卷积深度
        P %极化矩阵
        T %卷积矩阵
        lambda_offset
        llr_layer_vec
        bit_layer_vec
    end

    methods
        function obj = paccode(N,k,g,rate_profiling)
            %PACCODE 构造此类的实例
            %   此处显示详细说明
            n=ceil(log2(N));
            N=2^n;
            obj.N = N;
            obj.k = k;
            obj.g = g;
            obj.n = n;
            obj.R = k/N;
            obj.conv_depth = length(g);
            if(rate_profiling=='RM')
                obj.rate_profiling = RM_rate_profiling(obj);
            else
                error('Cannot find this rate profiling method.')
            end
            obj.P = get_P(obj.N);
            g_zp = [obj.g,zeros(1,obj.N-obj.conv_depth)];
            obj.T = triu(toeplitz(g_zp)); %upper-triangular Toeplitz matrix
            obj.lambda_offset = 2.^(0 : n);
            obj.llr_layer_vec = get_llr_layer(N);
            obj.bit_layer_vec = get_bit_layer(N);
        end

        function info_indices = RM_rate_profiling(obj)
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            Channel_indices=(0:obj.N-1)';
            bitStr=dec2bin(Channel_indices);
            bit=abs(bitStr)-48;
            RM_score=sum(bit,2);
            [~, sorted_indices]=sort(RM_score,'ascend');
            info_indices=sorted_indices(end-obj.k+1:end);
        end

        function x = encode(obj,d)
            if(length(d)~=obj.k)
                error('The length of the input d is not equal to k.')
            end
            % Rate Profile
            v=zeros(1,obj.N);
            v(obj.rate_profiling)=d;
            % convolutional encoder
            u=mod(v*obj.T,2);
            % Polar Encoding
            x=mod(u*obj.P,2);
        end




        % ---------------------------------------------------------------
        % Other Functions
        % ---------------------------------------------------------------

        % ---------------------------卷积码相关---------------------------

        function u = convTrans(v,c)
            u=zeros(1,length(v));
            curr_state = zeros(1,length(c)-1);
            for i = 1:length(v)
                [u(i),curr_state]=conv1bTrans(v(i),curr_state,c);
            end
        end

        function [u,next_state] = conv1bTrans(v,curr_state,c)
            u=mod(v*c(1),2);
            for j=2:length(c)
                if(c(j)==1)
                    u=mod(u+curr_state(j-1),2);
                end
            end
            next_state=[v,curr_state(1:end-1)];
        end

        % ---------------------------------------------------------------
        % -------------------------极化编码相关---------------------------
        function P = get_P(N)
            F = [1, 0 ; 1, 1];
            P = zeros(N, N);
            P(1 : 2, 1 : 2) = F;
            for i = 2 : log2(N)
                P(1 : 2^i, 1 : 2^i) = kron(P(1 : 2^(i - 1), 1 : 2^(i - 1)), F);
            end

        end
        % ---------------------------------------------------------------
        % -------------------------列表译码相关---------------------------
        function layer_vec = get_llr_layer(N)
            layer_vec = zeros(N , 1);
            for phi = 1 : N - 1
                psi = phi;
                layer = 0;
                while(mod(psi, 2) == 0)
                    psi = floor(psi/2);
                    layer = layer + 1;
                end
                layer_vec(phi + 1) = layer;
            end
        end

        function layer_vec = get_bit_layer(N)
            layer_vec = zeros(N, 1);
            for phi = 0 : N - 1
                psi = floor(phi/2);
                layer = 0;
                while(mod(psi, 2) == 1)
                    psi = floor(psi/2);
                    layer = layer + 1;
                end
                layer_vec(phi + 1) = layer;
            end
        end
        % ---------------------------------------------------------------
    end
end

