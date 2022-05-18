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
        GN %极化矩阵
        T %卷积矩阵
        lambda_offset %列表译码相关参数-'分段向量'
        llr_layer_vec %列表译码相关参数-'实际LLR计算执行层数'
        bit_layer_vec %列表译码相关参数-'实际比特返回层数'
        crc_length
        crc_parity_check
        H_crc
    end

    methods
        function obj = paccode(N,k,g,crc_length,rate_profiling,varargin)
            %PACCODE 构造此类的实例
            n=ceil(log2(N));
            N=2^n;
            obj.N = N;
            obj.k = k;
            obj.g = g;
            obj.n = n;
            obj.R = k/N;
            obj.crc_length = crc_length;
            obj.conv_depth = length(g);
            if(rate_profiling=='RM')
                obj.rate_profiling = RM_rate_profiling(obj);
            elseif(rate_profiling=='GA')
                if(size(varargin,2)>0)
                    dsnr = varargin{1};
                    obj.rate_profiling = GA_rate_profiling(obj,dsnr);
                else
                    error('You should input the design snr(dB).')
                end
            else
                error('Cannot find this rate profiling method.')
            end
            obj.GN = get_GN(obj.N);
            g_zp = [obj.g,zeros(1,obj.N-obj.conv_depth)];
            obj.T = triu(toeplitz(g_zp)); %upper-triangular Toeplitz matrix
            obj.lambda_offset = 2.^(0 : n);
            obj.llr_layer_vec = get_llr_layer(N);
            obj.bit_layer_vec = get_bit_layer(N);
            if(crc_length > 0)
                [~, ~, g_crc] = get_crc_objective(crc_length);
                [G_crc, obj.H_crc] = crc_generator_matrix(g_crc, k);
                obj.crc_parity_check = G_crc(:, k + 1 : end)';
            end

        end

        function info_indices = RM_rate_profiling(obj)
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            Channel_indices=(0:obj.N-1)';
            bitStr=dec2bin(Channel_indices);
            bit=abs(bitStr)-48;
            RM_score=sum(bit,2);
            [~, sorted_indices]=sort(RM_score,'ascend');
            info_indices=sort(sorted_indices(end-(obj.k+obj.crc_length)+1:end),'ascend');
        end

        function info_indices = GA_rate_profiling(obj,dsnr)
            %METHOD1 此处显示有关此方法的摘要
            %   此处显示详细说明
            sigma = 1/sqrt(2 * obj.R) * 10^(-dsnr/20);
            [channels, ~] = GA(sigma, obj.N);
            [~, channel_ordered] = sort(channels, 'descend');
            info_indices = sort(channel_ordered(1 : obj.k + obj.crc_length), 'ascend');
        end

        function [Pe] = get_PE_GA(obj,dsnr)
            %GET_PE 此处显示有关此函数的摘要
            %   此处显示详细说明
            sigma = 1/sqrt(2 * obj.R) * 10^(-dsnr/20);
            [channels, ~] = GA(sigma, obj.N);
            Pe = 1/2 * erfc(0.5*sqrt(channels));
        end



        function x = encode(obj,d)
            if(length(d)~=obj.k)
                error('The length of the input d is not equal to k.')
            end
            % Rate Profile
            if(obj.crc_length > 0)

                info_with_crc = [d; mod(obj.crc_parity_check * d, 2)];
            else
                info_with_crc = d;
            end
            v=zeros(1,obj.N);
            v(obj.rate_profiling) = info_with_crc;
            % convolutional encoder
            u=mod(v*obj.T,2);
            % Polar Encoding
            x=mod(u*obj.GN,2)';
        end

        function u_esti = SCL_decoder(obj,llr, L)
            %LLR-based SCL deocoder, a single function, no other sub-functions.
            %Frequently calling sub-functions will derease the efficiency of MATLAB
            %codes.
            %const
            N = obj.N;
            n = obj.n;
            K = obj.k;
            g = obj.g;
            llr_layer_vec = obj.llr_layer_vec;
            bit_layer_vec = obj.bit_layer_vec;
            lambda_offset = obj.lambda_offset;
            frozen_bits = ones(1,N);
            frozen_bits(obj.rate_profiling) = 0;

            %memory declared
            %If you can understand lazy copy and you just start learning polar codes
            %for just fews days, you are very clever,
            P = zeros(N - 1, L); %Channel llr is public-used, so N - 1 is enough.
            C = zeros(N - 1, 2 * L);%I do not esitimate (x1, x2, ... , xN), so N - 1 is enough.
            u = zeros(K, L);%unfrozen bits that polar codes carry, including crc bits.
            PM = zeros(L, 1);%Path metrics
            activepath = zeros(L, 1);%Indicate if the path is active. '1'→active; '0' otherwise.
            cnt_u = 1;%information bit counter
            %initialize
            activepath(1) = 1;
            %decoding starts
            %default: in the case of path clone, the origianl path always corresponds to bit 0, while the new path bit 1.
            curr_state = zeros(obj.conv_depth-1, L);
            curr_state_temp = zeros(obj.conv_depth-1, L);
            u_left=zeros(1,L);
            u_right=zeros(1,L);
            for phi = 0 : N - 1
                phi_mod_2 = mod(phi, 2);
                for l_index = 1 : L
                    if activepath(l_index) == 0
                        continue;
                    end
                    P(:,l_index)=update_P(obj,phi,P(:,l_index),C(:,2*l_index-1:2*l_index),llr);
                end
                if frozen_bits(phi + 1) == 0%if now we decode an unfrozen bit
                    PM_pair = realmax * ones(2, L);
                    for l_index = 1 : L
                        if activepath(l_index) == 0
                            continue;
                        end
                        curr_state_temp(:,l_index) = curr_state(:,l_index);
                        [u_left(l_index),curr_state(:,l_index)] = conv1bTrans(0,curr_state(:,l_index),g);
                        [u_right(l_index),curr_state_temp(:,l_index)] = conv1bTrans(1,curr_state_temp(:,l_index),g);
                        PM_pair(1, l_index) = calc_PM(PM(l_index),P(1, l_index),u_left(l_index));
                        PM_pair(2, l_index) = calc_PM(PM(l_index),P(1, l_index),u_right(l_index));
                    end
                    middle = min(2 * sum(activepath), L);
                    PM_sort = sort(PM_pair(:));
                    PM_cv = PM_sort(middle);
                    compare = PM_pair <= PM_cv;
                    kill_index = zeros(L, 1);%to record the index of the path that is killed
                    kill_cnt = 0;%the total number of killed path
                    %the above two variables consist of a stack
                    for i = 1 : L
                        if (compare(1, i) == 0)&&(compare(2, i) == 0)%which indicates that this path should be killed
                            activepath(i) = 0;
                            kill_cnt = kill_cnt + 1;%push stack
                            kill_index(kill_cnt) = i;
                        end
                    end
                    for l_index = 1 : L
                        if activepath(l_index) == 0
                            continue;
                        end
                        path_state = compare(1, l_index) * 2 + compare(2, l_index);
                        switch path_state%path_state can equal to 0, but in this case we do no operation.
                            case 1 % PM of the second row is lower
                                u(cnt_u, l_index) = 1;
                                C(:,2*l_index-1:2*l_index)=update_C(obj,phi,C(:,2*l_index-1:2*l_index),u_right(l_index));
                                PM(l_index) = PM_pair(2, l_index);
                                curr_state(:,l_index) = curr_state_temp(:,l_index);
                            case 2 % PM of the first row is lower
                                u(cnt_u, l_index) = 0;
                                C(:,2*l_index-1:2*l_index)=update_C(obj,phi,C(:,2*l_index-1:2*l_index),u_left(l_index));
                                PM(l_index) = PM_pair(1, l_index);
                            case 3 %
                                index = kill_index(kill_cnt);
                                kill_cnt = kill_cnt - 1;%pop stack
                                activepath(index) = 1;
                                %lazy copy
                                C(:, 2 * index - 1 : 2 * index) = C(:, 2 * l_index - 1 : 2 * l_index);
                                P(:, index) = P(:, l_index);
                                u(:, index) = u(:, l_index);
                                curr_state(:,index) = curr_state_temp(:,l_index);
                                u(cnt_u, l_index) = 0;
                                u(cnt_u, index) = 1;
                                C(:,2*l_index-1:2*l_index)=update_C(obj,phi,C(:,2*l_index-1:2*l_index),u_left(l_index));
                                C(:,2*index-1:2*index)=update_C(obj,phi,C(:,2*index-1:2*index),u_right(l_index));
                                PM(l_index) = PM_pair(1, l_index);
                                PM(index) = PM_pair(2, l_index);
                        end
                    end
                    cnt_u = cnt_u + 1;
                else%frozen bit operation
                    for l_index = 1 : L
                        if activepath(l_index) == 0
                            continue;
                        end
                        [u_temp,curr_state(:,l_index)] = conv1bTrans(0,curr_state(:,l_index),g);
                        PM(l_index)=calc_PM(PM(l_index),P(1, l_index),u_temp);

                        C(:,2*l_index-1:2*l_index)=update_C(obj,phi,C(:,2*l_index-1:2*l_index),u_temp);

                    end
                end


            end
            %path selection.
            %path selection.
            activepath=logical(activepath);
            PM_active = PM(activepath);
            u_active = u(:,activepath);
            [~, path_ordered] = sort(PM_active);
            if (obj.crc_length>0)
                for l_index = 1 : length(PM_active)
                    path_num = path_ordered(l_index);
                    info_with_crc = u(:, path_num);
                    err = sum(mod(obj.H_crc * info_with_crc, 2));
                    if err == 0
                        u_esti = u_active(1:end-obj.crc_length, path_num);
                        break;
                    else
                        if l_index == length(PM_active)
                            u_esti = u_active(1:end-obj.crc_length, path_ordered(1));
                        end
                    end
                end
            else
                u_esti = u_active(1:end-obj.crc_length, path_ordered(1));
            end
        end

        function d_esti = Fano_decoder(obj,llr,pe,Delta,Delta_q,i_bu,maxDiversions)
            N=obj.N;
            n=obj.n;
            K=obj.k;
            CS = generate_CS(obj.rate_profiling,n);
            c_state=zeros(obj.conv_depth-1,1);
            curr_state=zeros(obj.conv_depth-1,K); %0~k-1,1:|g|-1;
            i=0;
            j=0;
            Threshold=0;
            P = zeros(N - 1, 1);
            C = zeros(N - 1, 2);%I do not esitimate (x1, x2, ... , xN), so N - 1 is enough.
            B=sum(log2(1-pe));
            alphaq=1;
            onMainPath=true;
            isBackTracking=false;
            toDiverge=false;
            biasUpdated=false;

            j_stem=-1;
            jj=-1;
            info_set = obj.rate_profiling;
            frozen_bits = ones(1,N);
            frozen_bits(info_set) = 0;
            delta=zeros(K,1);
            delta_s=zeros(K,1);
            miu=zeros(N,1);
            miuu=zeros(N,1);
            miuuu=zeros(N,1);
            u_esti = zeros(N,1);
            v = zeros(N,1);

            while i < N
                P = update_P(obj,i,P,C,llr);
                if frozen_bits(i+1) == 1
                    [u_esti(i+1),c_state] = conv1bTrans(0,c_state,obj.g);
                    if i==0
                        miu(i+1) =  B + m_func(P(1),u_esti(i+1))-alphaq*log2(1-pe(j+1));
                    else
                        miu(i+1) =  miu(i) + m_func(P(1),u_esti(i+1))-alphaq*log2(1-pe(j+1));
                    end
                    C = update_C(obj,i,C,u_esti(i+1));
                    i = i + 1;
                else
                    [u_left,c_state_left] = conv1bTrans(0,c_state,obj.g);
                    [u_right,c_state_right] = conv1bTrans(1,c_state,obj.g);
                    if i==0
                        miu_left =  B + m_func(P(1),u_left)-alphaq*log2(1-pe(j+1));
                        miu_right =  B + m_func(P(1),u_right)-alphaq*log2(1-pe(j+1));
                    else
                        miu_left =  miu(i) + m_func(P(1),u_left)-alphaq*log2(1-pe(j+1));
                        miu_right =  miu(i) + m_func(P(1),u_right)-alphaq*log2(1-pe(j+1));
                    end
                    if miu_left>miu_right
                        miu_max = miu_left;
                        miu_min = miu_right;
                        v_max = 0;
                        v_min = 1;
                    else
                        miu_max = miu_right;
                        miu_min = miu_left;
                        v_max = 1;
                        v_min = 0;
                    end

                    if onMainPath==true && isBackTracking == true
                        if miu_min>Threshold && CS(j+1)==1 && j<j_end
                            onMainPath=false;
                            delta_s(j+1)=1;
                            j_stem=j;
                            P_s = P;
                            C_s = C;
                        elseif j==j_end
                            isBackTracking = false;
                            Threshold = floor(miu_end/Delta)*Delta;
                        end
                    end

                    if miu_max > Threshold
                        if toDiverge == false
                            v(i+1)=v_max;
                            if v_max == 0
                                u_esti(i+1)=u_left;
                            else
                                u_esti(i+1)=u_right;
                            end
                            if onMainPath == true && delta_s(j+1)==1
                                miu(i+1)=miu_max;
                                miuu(i+1)=miu_min;
                            else
                                miu(i+1)=miu_max;
                                miuu(i+1)=miuuu(i+1);
                            end
                            delta(j+1)=0;
                        else
                            v(i+1)=v_min;
                            if v_min == 0
                                u_esti(i+1)=u_left;
                            else
                                u_esti(i+1)=u_right;
                            end
                            miu(i+1)=miu_min;
                            miuu(i+1)=miu_max;
                            delta(j+1)=1;
                            toDiverge=false;
                        end
                        curr_state(:,j+1)=c_state;
                        if v(i+1)==0
                            c_state=c_state_left;
                        else
                            c_state=c_state_right;
                        end
                        C = update_C(obj,i,C,u_esti(i+1));
                        i = i+1;
                        j = j+1;
                    else
                        if biasUpdated == false && i<i_bu
                            Threshold=floor(miu_max/Delta)*Delta;
                        elseif biasUpdated == false && i>=i_bu
                            if miu_max < B
                                alphaq=ceil(miu_max/(B*Delta_q))*Delta_q;
                                biasUpdated = true;
                                for k=0:j
                                    miuu(info_set(k+1))=miuu(info_set(k+1))+(alphaq-1)*(B-sum(log2(1-pe(1:info_set(k+1)))));
                                end
                                miu(info_set(1)-1) = miu(info_set(1)-1) +  (alphaq-1)*(B-sum(log2(1-pe(1:info_set(1)-1))));
                                miu(info_set(j+1)-1) = miu(info_set(j+1)-1) +  (alphaq-1)*(B-sum(log2(1-pe(1:info_set(j+1)-1))));
                            end
                        end
                        curr_state(:,j+1) = c_state;
                        if onMainPath == false
                            if miuuu(info_set(j_stem+1)) < miu_max
                                miuuu(info_set(j_stem+1)) = miu_max;
                            end
                        else
                            j_end = j;
                            miu_end = miu_max;
                            frmMAINpath = true;
                            isBackTracking = true;
                        end
                        %% Moving Back Start
                        isMovingBack = false;
                        breaknreturn = false;
                        while true
                            jj=j;
                            if frmMAINpath == true
                                for k=0:jj-1
                                    if miuu(info_set(k+1))>Threshold && CS(k+1)==1
                                        jj=k;
                                        j_stem=k;
                                        isMovingBack=true;
                                        P_s = P;
                                        C_s = C;
                                        break;
                                    end
                                end

                                if jj==j
                                    toDiverge = false;
                                    break;
                                end
                            else
                                for k = jj-1 : -1 : 0
                                    if j_stem == k
                                        jj=k;
                                        P=P_s;
                                        C=C_s;
                                        toDiverge=false;
                                        breaknreturn = true;
                                        break;
                                    end
                                    if miuu(info_set(k+1)) > Threshold && CS(k+1) == 1
                                        if sum(delta(1:k+1))>= maxDiversions
                                            continue;
                                        end
                                        if(delta(k+1)==1)
                                            jj=k;
                                            isMovingBack=true;
                                            break;
                                        end
                                    end
                                end
                            end

                            if breaknreturn
                                break;
                            end

                            if isMovingBack == true
                                i_cur=info_set(j+1)-1;
                                i_start = info_set(jj+1)-1;
                                [P,C]= updateLLRsPSs(obj,i_start,i_cur,u_esti,P,C,llr);
                                if(delta(jj+1)==0)
                                    toDiverge = true;
                                    break;
                                elseif jj==0
                                    toDiverge=false;
                                    break;
                                end
                            end


                        end

                        % Moving Back End
                        if toDiverge == false && (jj == j_stem || jj==j)
                            onMainPath = true;
                        else
                            onMainPath = false;
                        end
                        i = info_set(jj+1)-1;
                        j=jj;
                        frmMAINpath=false;
                        c_state = curr_state(:,j+1);
                    end

                end
            end
            d_esti = v(info_set);

        end

        function [P,C]=updateLLRsPSs(obj,i_start,i_cur,u_esti,P,C,llr)
            if mod(i_cur,2) ~= 0
                i_cur=i_cur-1;
            end
            if mod(i_start,2) ~= 0
                i_start=i_start-1;
            end
            s_start = ffs(i_start,obj.n);
            s_max = smax(i_start,i_cur,obj.n);

            if s_start<=s_max
                i_minus1 = find_sMaxPos(s_start,s_max,i_start,obj.n);
                for i = i_minus1 : i_start
                    P = update_P(obj,i,P,C,llr);
                    C = update_C(obj,i,C,u_esti(i+1));
                end
            else
                P = update_P(obj,i_start,P,C);
            end
        end

        function C = updatePSBack(obj,i_minus1,s_max,u_esti,C)
            k=2^s_max;
            for i = i_minus1 + 1 - k : i_minus1
                C = update_C(obj,i,C,u_esti(i+1));
            end
        end


        function P = update_P(obj,phi,P,C,llr)
            if((phi == 0 || phi==obj.N/2) && ~exist('llr'))
                error("Need channel LLR to update P.")
                return;
            end

            N=obj.N;
            n=obj.n;
            lambda_offset=obj.lambda_offset;
            llr_layer_vec=obj.llr_layer_vec;
            layer = llr_layer_vec(phi + 1);

            switch phi%Decoding bits u_0 and u_N/2 needs channel LLR, so the decoding of them is separated from other bits.
                case 0
                    index_1 = lambda_offset(n);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1) = sign(llr(beta + 1)) * sign(llr(beta + index_1 + 1)) * min(abs(llr(beta + 1)), abs(llr(beta + index_1 + 1)));
                    end
                    for i_layer = n - 2 : -1 : 0
                        index_1 = lambda_offset(i_layer + 1);
                        index_2 = lambda_offset(i_layer + 2);
                        for beta = 0 : index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) *...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)), abs(P(beta + index_1 + index_2)));
                        end
                    end
                case N/2
                    index_1 = lambda_offset(n);
                    for beta = 0 : index_1 - 1
                        x_tmp = C(beta + index_1, 1);
                        P(beta + index_1) = (1 - 2 * x_tmp) * llr(beta + 1) + llr(beta + 1 + index_1);
                    end
                    for i_layer = n - 2 : -1 : 0
                        index_1 = lambda_offset(i_layer + 1);
                        index_2 = lambda_offset(i_layer + 2);
                        for beta = 0 : index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) *...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)), abs(P(beta + index_1 + index_2)));
                        end
                    end
                otherwise
                    index_1 = lambda_offset(layer + 1);
                    index_2 = lambda_offset(layer + 2);
                    for beta = 0 : index_1 - 1
                        P(beta + index_1) = (1 - 2 * C(beta + index_1, 1)) * P(beta + index_2) +...
                            P(beta + index_1 + index_2);
                    end
                    for i_layer = layer - 1 : -1 : 0
                        index_1 = lambda_offset(i_layer + 1);
                        index_2 = lambda_offset(i_layer + 2);
                        for beta = 0 : index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) *...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)),...
                                abs(P(beta + index_1 + index_2)));
                        end
                    end
            end
        end

        function C = update_C(obj,phi,C,u)
            N=obj.N;
            bit_layer_vec=obj.bit_layer_vec;
            lambda_offset=obj.lambda_offset;
            phi_mod_2 = mod(phi, 2);
            C(1, 1+phi_mod_2) = u;
            if (phi_mod_2  == 1) && (phi ~= N - 1)
                layer = bit_layer_vec(phi + 1);
                for i_layer = 0 : layer - 1
                    index_1 = lambda_offset(i_layer + 1);
                    index_2 = lambda_offset(i_layer + 2);
                    for beta = index_1 : 2 * index_1 - 1
                        C(beta + index_1, 2) = mod(C(beta, 1) + C(beta, 2), 2);%Left Column lazy copy
                        C(beta + index_2, 2) = C(beta, 2);
                    end
                end
                index_1 = lambda_offset(layer + 1);
                index_2 = lambda_offset(layer + 2);
                for beta = index_1 : 2 * index_1 - 1
                    C(beta + index_1, 1) = mod(C(beta, 1) + C(beta, 2), 2);%Left Column lazy copy
                    C(beta + index_2, 1) = C(beta, 2);
                end
            end
        end
    end
end


