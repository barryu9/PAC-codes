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
            obj.GN = get_GN(obj.N);
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
            info_indices=sort(sorted_indices(end-obj.k+1:end),'ascend');
        end

        function x = encode(obj,u)
            if(length(u)~=obj.k)
                error('The length of the input d is not equal to k.')
            end
            % Rate Profile
            v=zeros(1,obj.N);
            v(obj.rate_profiling) = u;
            % convolutional encoder
            u=mod(v*obj.T,2);
            % Polar Encoding
            x=mod(u*obj.GN,2)';
        end

        function polar_info_esti = SCL_decoder(obj,llr, L)
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
            lazy_copy = zeros(n, L);%If Lazy Copy is used, there is no data-copy in the decoding process. We only need to record where the data come from. Here,data refer to LLRs and partial sums.
            %Lazy Copy is a relatively sophisticated operation for new learners of polar codes. If you do not understand such operation, you can directly copy data.
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
            lazy_copy(:, 1) = 1;
            %decoding starts
            %default: in the case of path clone, the origianl path always corresponds to bit 0, while the new path bit 1.
            curr_state = zeros(obj.conv_depth-1, L);
            curr_state_temp = zeros(obj.conv_depth-1, L);
            u_left=zeros(1,L);
            u_right=zeros(1,L);
            for phi = 0 : N - 1
                layer = llr_layer_vec(phi + 1);
                phi_mod_2 = mod(phi, 2);
                for l_index = 1 : L
                    if activepath(l_index) == 0
                        continue;
                    end
                    switch phi%Decoding bits u_0 and u_N/2 needs channel LLR, so the decoding of them is separated from other bits.
                        case 0
                            index_1 = lambda_offset(n);
                            for beta = 0 : index_1 - 1
                                P(beta + index_1, l_index) = sign(llr(beta + 1)) * sign(llr(beta + index_1 + 1)) * min(abs(llr(beta + 1)), abs(llr(beta + index_1 + 1)));
                            end
                            for i_layer = n - 2 : -1 : 0
                                index_1 = lambda_offset(i_layer + 1);
                                index_2 = lambda_offset(i_layer + 2);
                                for beta = 0 : index_1 - 1
                                    P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                                        sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                                end
                            end
                        case N/2
                            index_1 = lambda_offset(n);
                            for beta = 0 : index_1 - 1
                                x_tmp = C(beta + index_1, 2 * l_index - 1);
                                P(beta + index_1, l_index) = (1 - 2 * x_tmp) * llr(beta + 1) + llr(beta + 1 + index_1);
                            end
                            for i_layer = n - 2 : -1 : 0
                                index_1 = lambda_offset(i_layer + 1);
                                index_2 = lambda_offset(i_layer + 2);
                                for beta = 0 : index_1 - 1
                                    P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                                        sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)), abs(P(beta + index_1 + index_2, l_index)));
                                end
                            end
                        otherwise
                            index_1 = lambda_offset(layer + 1);
                            index_2 = lambda_offset(layer + 2);
                            for beta = 0 : index_1 - 1
                                P(beta + index_1, l_index) = (1 - 2 * C(beta + index_1, 2 * l_index - 1)) * P(beta + index_2, lazy_copy(layer + 2, l_index)) +...
                                    P(beta + index_1 + index_2, lazy_copy(layer + 2, l_index));
                            end
                            for i_layer = layer - 1 : -1 : 0
                                index_1 = lambda_offset(i_layer + 1);
                                index_2 = lambda_offset(i_layer + 2);
                                for beta = 0 : index_1 - 1
                                    P(beta + index_1, l_index) = sign(P(beta + index_2, l_index)) *...
                                        sign(P(beta + index_1 + index_2, l_index)) * min(abs(P(beta + index_2, l_index)),...
                                        abs(P(beta + index_1 + index_2, l_index)));
                                end
                            end
                    end
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
                                C(1, 2 * l_index - 1 + phi_mod_2) = u_right(l_index);
                                PM(l_index) = PM_pair(2, l_index);
                                curr_state(:,l_index) = curr_state_temp(:,l_index);
                            case 2 % PM of the first row is lower
                                u(cnt_u, l_index) = 0;
                                C(1, 2 * l_index - 1 + phi_mod_2) = u_left(l_index);
                                PM(l_index) = PM_pair(1, l_index);
                            case 3 %
                                index = kill_index(kill_cnt);
                                kill_cnt = kill_cnt - 1;%pop stack
                                activepath(index) = 1;
                                %lazy copy
                                lazy_copy(:, index) = lazy_copy(:, l_index);
                                u(:, index) = u(:, l_index);
                                curr_state(:,index) = curr_state_temp(:,l_index);

                                u(cnt_u, l_index) = 0;
                                u(cnt_u, index) = 1;
                                C(1, 2 * l_index - 1 + phi_mod_2) = u_left(l_index);
                                C(1, 2 * index - 1 + phi_mod_2) = u_right(l_index);
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

                        if phi_mod_2 == 0
                            C(1, 2 * l_index - 1) = u_temp;
                        else
                            C(1, 2 * l_index) = u_temp;
                        end
                    end
                end

                for l_index = 1 : L%partial-sum return
                    if activepath(l_index) == 0
                        continue
                    end
                    if (phi_mod_2  == 1) && (phi ~= N - 1)
                        layer = bit_layer_vec(phi + 1);
                        for i_layer = 0 : layer - 1
                            index_1 = lambda_offset(i_layer + 1);
                            index_2 = lambda_offset(i_layer + 2);
                            for beta = index_1 : 2 * index_1 - 1
                                C(beta + index_1, 2 * l_index) = mod(C(beta, 2 *  lazy_copy(i_layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                                C(beta + index_2, 2 * l_index) = C(beta, 2 * l_index);
                            end
                        end
                        index_1 = lambda_offset(layer + 1);
                        index_2 = lambda_offset(layer + 2);
                        for beta = index_1 : 2 * index_1 - 1
                            C(beta + index_1, 2 * l_index - 1) = mod(C(beta, 2 * lazy_copy(layer + 1, l_index) - 1) + C(beta, 2 * l_index), 2);%Left Column lazy copy
                            C(beta + index_2, 2 * l_index - 1) = C(beta, 2 * l_index);
                        end
                    end
                end
                %lazy copy
                if phi < N - 1
                    for i_layer = 1 : llr_layer_vec(phi + 2) + 1
                        for l_index = 1 : L
                            lazy_copy(i_layer, l_index) = l_index;
                        end
                    end
                end
            end
            %path selection.
            activepath=logical(activepath);
            PM_active = PM(activepath);
            u_active = u(:,activepath);
            [~, path_ordered] = sort(PM_active);

            polar_info_esti = u_active(:, path_ordered(1));
        end
    end
end

