classdef paccode
    %PACCODE 定义一个paccode类
    properties
        code_length %码长
        information_length %信息长度
        gen %卷积生成序列
        rate %码率
        rate_profiling %码率分布
        conv_depth %卷积深度
        GN %极化矩阵
        convolution_matrix %卷积矩阵
        lambda_offset %列表译码相关参数-'分段向量'
        llr_layer_vec %列表译码相关参数-'实际LLR计算执行层数'
        bit_layer_vec %列表译码相关参数-'实际比特返回层数'
        crc_length
        crc_parity_check
        H_crc
        llr
    end

    methods
        function obj = paccode(N, k, g, crc_length, rate_profiling, varargin)
            %PACCODE 构造此类的实例
            n = ceil(log2(N));
            N = 2^n;

            obj.code_length = N;
            obj.information_length = k;
            obj.gen = g;
            obj.rate = k / N;

            obj.crc_length = crc_length;
            obj.conv_depth = length(g);
            global operation_count_C;
            global operation_count_P;

            if (strcmp(rate_profiling, 'RM'))
                obj.rate_profiling = RM_rate_profiling(obj);
            elseif (strcmp(rate_profiling, 'GA'))
                if (size(varargin, 2) > 0)
                    dsnr = varargin{1};
                    obj.rate_profiling = GA_rate_profiling(obj, dsnr);
                else
                    error('You should input the design snr (dB).')
                end
            elseif (strcmp(rate_profiling, 'RM-Polar'))
                if (size(varargin, 2) > 0)
                    dsnr = varargin{1};
                    obj.rate_profiling = RM_Polar_rate_profiling(obj, dsnr);
                else
                    error('You should input the design snr (dB).')
                end
            else
                error('Cannot find this rate profiling method.')
            end

            obj.GN = get_GN(obj.code_length);
            g_zp = [obj.gen, zeros(1, obj.code_length-obj.conv_depth)];
            obj.convolution_matrix = triu(toeplitz(g_zp)); %upper-triangular Toeplitz matrix

            obj.lambda_offset = 2.^(0:n);
            obj.llr_layer_vec = get_llr_layer(N);
            obj.bit_layer_vec = get_bit_layer(N);

            if (crc_length > 0)
                [~, ~, g_crc] = get_crc_objective(crc_length);
                [G_crc, obj.H_crc] = crc_generator_matrix(g_crc, k);
                obj.crc_parity_check = G_crc(:, k+1:end)';
            end


        end

        function info_indices = RM_rate_profiling(obj)
            %RM_rate_profiling RM码率分配
            %根据信道序号对应的二进制表示中'1'的个数进行排序，从高到低选出信息信道
            Channel_indices = (0:obj.code_length - 1)';
            bitStr = dec2bin(Channel_indices);
            bit = abs(bitStr) - 48;
            RM_score = sum(bit, 2);
            [~, sorted_indices] = sort(RM_score, 'ascend');
            info_indices = sort(sorted_indices(end-(obj.information_length + obj.crc_length)+1:end), 'ascend');
        end

        function [info_indices,info_indices_revorder] = RM_Polar_rate_profiling(obj, dsnr)
            %RM_Polar_rate_profiling RM-Polar码率构造
            %使用GA算法辅助，也可以使用BEC巴氏参数等效法
            %dsnr是GA的design snr（以dB为单位）
            n = log2(obj.code_length);
            Channel_indices = (0:obj.code_length - 1)';
            bitStr = dec2bin(Channel_indices);
            bit = abs(bitStr) - 48;
            RM_score = sum(bit, 2);
            sigma = 1 / sqrt(2*obj.rate) * 10^(-dsnr / 20);
            [channel_score, ~] = GA(sigma, obj.code_length);
            mask = zeros(obj.code_length, 4);
            for i = 1:obj.code_length
                mask(i, :) = [i - 1, RM_score(i), channel_score(i), 1];
            end
            weightCount = zeros(n+1, 1);
            for i = 1:obj.code_length
                weightCount(RM_score(i)+1) = weightCount(RM_score(i)+1) + 1;
            end
            bitCnt = 0;
            k = 1;
            while bitCnt + weightCount(k) <= obj.code_length - (obj.information_length + obj.crc_length)
                for i = 1:obj.code_length
                    if RM_score(i) == k - 1
                        mask(i, 4) = 0;
                        bitCnt = bitCnt + 1;
                    end
                end
                k = k + 1;
            end
            mask2 = [];
            for i = 1:obj.code_length
                if mask(i, 2) == k - 1
                    mask2 = [mask2; mask(i, :)];
                end
            end
            [~, mask2_sorted_indices] = sort(mask2(:, 3), 'ascend');
            mask2 = mask2(mask2_sorted_indices, :);
            remainder = obj.code_length - (obj.information_length + obj.crc_length) - bitCnt;
            for i = 1:remainder
                mask(mask2(i, 1)+1, 4) = 0;
            end
            acceptable_bits = logical(mask(:, 4));
%             acceptable_bits=bitrevorder(acceptable_bits);
            mask3 = mask(acceptable_bits,:);
            [~, mask3_sorted_indices] = sort(mask3(:, 2), 'descend');
            mask3 = mask3(mask3_sorted_indices, :);
            mask3(1:obj.information_length + obj.crc_length,1)=bitrevorder(mask3(1:obj.information_length + obj.crc_length,1));
            info_indices = sort(mask3(1:obj.information_length + obj.crc_length,1), 'ascend')' + 1;

        end

        function  [info_indices,info_indices_revorder] = GA_rate_profiling(obj, dsnr)
            %GA_rate_profiling GA码率分配
            %dsnr是GA的design snr（以dB为单位）
            sigma = 1 / sqrt(2*obj.rate) * 10^(-dsnr / 20);
            [channels, ~] = GA(sigma, obj.code_length);
            [~, channel_ordered] = sort(channels, 'descend');
            info_indices = sort(channel_ordered(1:obj.information_length+obj.crc_length), 'ascend');
        end

        function [Pe] = get_PE_GA(obj, dsnr)
            %get_PE_GA 获取各子信道错误概率
            %使用GA估计各子信道的错误概率，Fano译码器中使用
            %dsnr是GA的design snr（以dB为单位）
            sigma = 1 / sqrt(2*obj.rate) * 10^(-dsnr / 20);
            [channels, ~] = GA(sigma, obj.code_length);
            Pe = 1 / 2 * erfc(0.5*sqrt(channels));
        end

        function x = encode(obj, d)
            %encode PAC编码
            %将输入的源信息d编码成PAC码
            if (length(d) ~= obj.information_length)
                error('The length of the input d is not equal to k.')
            end
            % Rate Profile
            if (obj.crc_length > 0)
                info_with_crc = [d; mod(obj.crc_parity_check*d, 2)];
            else
                info_with_crc = d;
            end
            v = zeros(1, obj.code_length);
            v(obj.rate_profiling) = info_with_crc;
            % convolutional encoder
            u = mod(v*obj.convolution_matrix, 2);
            % Polar Encoding
            x = mod(u*obj.GN, 2)';
        end

        function d_esti = SCL_decoder(obj, llr, L)
            %LLR-based SCL deocoder, a single function, no other sub-functions.
            %Frequently calling sub-functions will derease the efficiency of MATLAB
            %codes.
            %const
            N = obj.code_length;
            K = obj.information_length;
            g = obj.gen;
            obj.llr = llr;
            frozen_bits = ones(1, N);
            frozen_bits(obj.rate_profiling) = 0;
            %memory declared
            %If you can understand lazy copy and you just start learning polar codes
            %for just fews days, you are very clever,
            P = zeros(N - 1, L); %Channel llr is public-used, so N - 1 is enough.
            C = zeros(N - 1, 2 * L); %I do not esitimate (x1, x2, ... , xN), so N - 1 is enough.
            d = zeros(K, L); %unfrozen bits that polar codes carry, including crc bits.
            PM = zeros(L, 1); %Path metrics
            activepath = zeros(L, 1); %Indicate if the path is active. '1'→active; '0' otherwise.
            cnt_u = 1; %information bit counter
            %initialize
            activepath(1) = 1;
            %decoding starts
            %default: in the case of path clone, the origianl path always corresponds to bit 0, while the new path bit 1.
            curr_state = zeros(obj.conv_depth-1, L);
            curr_state_temp = zeros(obj.conv_depth-1, L);
            u_left = zeros(1, L);
            u_right = zeros(1, L);
            for phi = 0:N - 1
                for l_index = 1:L
                    if activepath(l_index) == 0
                        continue;
                    end
                    P(:, l_index) = update_P(obj, phi, P(:, l_index), C(:, 2*l_index-1:2*l_index));
                end
                if frozen_bits(phi + 1) == 0 %if now we decode an unfrozen bit
                    PM_pair = realmax * ones(2, L);
                    for l_index = 1:L
                        if activepath(l_index) == 0
                            continue;
                        end
                        curr_state_temp(:, l_index) = curr_state(:, l_index);
                        [u_left(l_index), curr_state(:, l_index)] = conv1bTrans(0, curr_state(:, l_index), g);
                        [u_right(l_index), curr_state_temp(:, l_index)] = conv1bTrans(1, curr_state_temp(:, l_index), g);
                        PM_pair(1, l_index) = calc_PM(PM(l_index), P(1, l_index), u_left(l_index));
                        PM_pair(2, l_index) = calc_PM(PM(l_index), P(1, l_index), u_right(l_index));
                    end
                    middle = min(2*sum(activepath), L);
                    PM_sort = sort(PM_pair(:));
                    PM_cv = PM_sort(middle);
                    compare = PM_pair <= PM_cv;
                    kill_index = zeros(L, 1); %to record the index of the path that is killed
                    kill_cnt = 0; %the total number of killed path
                    %the above two variables consist of a stack
                    for i = 1:L
                        if (compare(1, i) == 0) && (compare(2, i) == 0) %which indicates that this path should be killed
                            activepath(i) = 0;
                            kill_cnt = kill_cnt + 1; %push stack
                            kill_index(kill_cnt) = i;
                        end
                    end
                    for l_index = 1:L
                        if activepath(l_index) == 0
                            continue;
                        end
                        path_state = compare(1, l_index) * 2 + compare(2, l_index);
                        switch path_state %path_state can equal to 0, but in this case we do no operation.
                            case 1 % PM of the second row is lower
                                d(cnt_u, l_index) = 1;
                                C(:, 2*l_index-1:2*l_index) = update_C(obj, phi, C(:, 2*l_index-1:2*l_index), u_right(l_index));
                                PM(l_index) = PM_pair(2, l_index);
                                curr_state(:, l_index) = curr_state_temp(:, l_index);
                            case 2 % PM of the first row is lower
                                d(cnt_u, l_index) = 0;
                                C(:, 2*l_index-1:2*l_index) = update_C(obj, phi, C(:, 2*l_index-1:2*l_index), u_left(l_index));
                                PM(l_index) = PM_pair(1, l_index);
                            case 3 %
                                index = kill_index(kill_cnt);
                                kill_cnt = kill_cnt - 1; %pop stack
                                activepath(index) = 1;
                                %lazy copy
                                C(:, 2*index-1:2*index) = C(:, 2*l_index-1:2*l_index);
                                P(:, index) = P(:, l_index);
                                d(:, index) = d(:, l_index);
                                curr_state(:, index) = curr_state_temp(:, l_index);
                                d(cnt_u, l_index) = 0;
                                d(cnt_u, index) = 1;
                                C(:, 2*l_index-1:2*l_index) = update_C(obj, phi, C(:, 2*l_index-1:2*l_index), u_left(l_index));
                                C(:, 2*index-1:2*index) = update_C(obj, phi, C(:, 2*index-1:2*index), u_right(l_index));
                                PM(l_index) = PM_pair(1, l_index);
                                PM(index) = PM_pair(2, l_index);
                        end
                    end
                    cnt_u = cnt_u + 1;
                else %frozen bit operation
                    for l_index = 1:L
                        if activepath(l_index) == 0
                            continue;
                        end
                        [u_temp, curr_state(:, l_index)] = conv1bTrans(0, curr_state(:, l_index), g);
                        PM(l_index) = calc_PM(PM(l_index), P(1, l_index), u_temp);
                        C(:, 2*l_index-1:2*l_index) = update_C(obj, phi, C(:, 2*l_index-1:2*l_index), u_temp);

                    end
                end


            end
            %path selection.
            activepath = logical(activepath);
            PM_active = PM(activepath);
            d_active = d(:, activepath);
            [~, path_ordered] = sort(PM_active);
            if (obj.crc_length > 0)
                for l_index = 1:length(PM_active)
                    path_num = path_ordered(l_index);
                    info_with_crc = d(:, path_num);
                    err = sum(mod(obj.H_crc*info_with_crc, 2));
                    if err == 0
                        d_esti = d_active(1:end-obj.crc_length, path_num);
                        break;
                    else
                        if l_index == length(PM_active)
                            d_esti = d_active(1:end-obj.crc_length, path_ordered(1));
                        end
                    end
                end
            else
                d_esti = d_active(1:end-obj.crc_length, path_ordered(1));
            end
        end

        function [d_esti] = Fano_decoder(obj, llr, pe, Delta)
            % Init;
            c_state = zeros(obj.conv_depth-1, 1);
            curr_state = zeros(obj.conv_depth-1, obj.information_length); %0~k-1,1:|g|-1;
            i = 0;
            j = 0;
            obj.llr = llr;
            Threshold = 0;
            P = zeros(obj.code_length-1, 1);
            C = zeros(obj.code_length-1, 2);
            B = sum(log2(1-pe));
            alphaq = 1;
            info_set = obj.rate_profiling;
            frozen_bits = ones(1, obj.code_length);
            frozen_bits(info_set) = 0;
            delta = zeros(obj.information_length, 1);
            mu = zeros(obj.code_length, 1);
            bmetric = zeros(obj.information_length, 1);
            bmetric_cut = zeros(obj.information_length, 1);
            u_esti = zeros(obj.code_length, 1);
            v_esti = zeros(obj.code_length, 1);
            visited_before = false;
            %while not end of tree do
            while i < obj.code_length
                P = update_P(obj, i, P, C);
                if frozen_bits(i+1) == 1
                    [u_esti(i+1), c_state] = conv1bTrans(0, c_state, obj.gen);
                    if i == 0
                        mu(i+1) = B + m_func(P(1), u_esti(i+1)) - alphaq * log2(1-pe(i+1));
                    else
                        mu(i+1) = mu(i) + m_func(P(1), u_esti(i+1)) - alphaq * log2(1-pe(i+1));
                    end
                    curr_state(:, j+1) = c_state;
                    C = update_C(obj, i, C, u_esti(i+1));
                    i = i + 1;
                else

                    % look forward to best node
                    [u_left, c_state_left] = conv1bTrans(0, c_state, obj.gen);
                    [u_right, c_state_right] = conv1bTrans(1, c_state, obj.gen);
                    if i == 0
                        mu_left = B + m_func(P(1), u_left) - alphaq * log2(1-pe(i+1));
                        mu_right = B + m_func(P(1), u_right) - alphaq * log2(1-pe(i+1));
                    else
                        mu_left = mu(i) + m_func(P(1), u_left) - alphaq * log2(1-pe(i+1));
                        mu_right = mu(i) + m_func(P(1), u_right) - alphaq * log2(1-pe(i+1));
                    end
                    if mu_left > mu_right
                        mu_max = mu_left;
                        mu_min = mu_right;

                    else
                        mu_max = mu_right;
                        mu_min = mu_left;
                    end

                    if mu_max >= Threshold
                        if visited_before == false
                            if mu_left > mu_right
                                v_esti(i+1) = 0;
                                u_esti(i+1) = u_left;
                            else
                                v_esti(i+1) = 1;
                                u_esti(i+1) = u_right;
                            end
                            bmetric(j+1) = mu_max;
                            bmetric_cut(j+1) = mu_min;
                            mu(i+1) = mu_max;
                            delta(j+1) = 0;
                            C = update_C(obj, i, C, u_esti(i+1));
                            curr_state(:, j+1) = c_state;
                            if (v_esti(i+1) == 0)
                                c_state = c_state_left;
                            else
                                c_state = c_state_right;
                            end
                            i = i + 1;
                            j = j + 1;
                        else
                            if mu_min > Threshold
                                if mu_left < mu_right
                                    v_esti(i+1) = 0;
                                    u_esti(i+1) = u_left;
                                else
                                    v_esti(i+1) = 1;
                                    u_esti(i+1) = u_right;
                                end
                                bmetric(j+1) = mu_min;
                                bmetric_cut(j+1) = mu_max;
                                mu(i+1) = mu_min;
                                C = update_C(obj, i, C, u_esti(i+1));
                                curr_state(:, j+1) = c_state;
                                if (v_esti(i+1) == 0)
                                    c_state = c_state_left;
                                else
                                    c_state = c_state_right;
                                end
                                delta(j+1) = 1;
                                i = i + 1;
                                j = j + 1;
                                visited_before = false;
                            else
                                if j == 0
                                    Threshold = Threshold - Delta;
                                    visited_before = false;
                                else
                                    curr_state(:, j+1) = c_state;
                                    [Threshold, j, visited_before, P, C] = moveback(obj, bmetric, bmetric_cut, j, Threshold, delta, u_esti, P, C, llr, Delta);
                                    if j == 0
                                        c_state = zeros(obj.conv_depth-1, 1);
                                    else
                                        c_state = curr_state(:, j+1);
                                    end
                                    i = obj.rate_profiling(j+1) - 1;
                                end
                            end
                        end
                    else
                        if j == 0
                            while mu_max < Threshold
                                Threshold = Threshold - Delta;
                            end
                            visited_before = false;
                        else
                            curr_state(:, j+1) = c_state;
                            [Threshold, j, visited_before, P, C] = moveback(obj, bmetric, bmetric_cut, j, Threshold, delta, u_esti, P, C, llr, Delta);
                            if j == 0
                                c_state = zeros(obj.conv_depth-1, 1);
                            else
                                c_state = curr_state(:, j+1);
                            end
                            i = obj.rate_profiling(j+1) - 1;
                        end
                    end
                end
            end
            d_esti = v_esti(info_set);
        end

        function [Threshold, jj, visited_before, P, C] = moveback(obj, bmetric, bmetric_cut, j, Threshold, delta, u_esti, P, C, llr, Delta)
            while 1
                follow_other_branch = false;
                if j >= 1
                    mu_pre = bmetric(j);
                else
                    mu_pre = 0;
                end

                jj = j;
                for k = j - 1:-1:0
                    mu_pre = bmetric(k+1);
                    if mu_pre >= Threshold
                        if bmetric_cut(k+1) >= Threshold
                            jj = k;
                            tmp = bmetric(k+1);
                            bmetric(k+1) = bmetric_cut(k+1);
                            bmetric_cut(k+1) = tmp;
                            if delta(k+1) == 0
                                follow_other_branch = true;
                                break;
                            elseif k == 0
                                follow_other_branch = true;
                                jj = 0;
                                break;
                            end
                        end
                    end
                    if k == 0
                        mu_pre = -100;
                    end
                end

                i_cur = obj.rate_profiling(j+1) - 1;
                if (mu_pre >= Threshold) && (jj ~= 0 || follow_other_branch == true)
                    i_start = obj.rate_profiling(jj+1) - 1;
                    [P, C] = updateLLRsPSs(obj, i_start, i_cur, u_esti, P, C);

                    if delta(jj+1) == 0
                        visited_before = true;
                        return;
                    elseif jj == 0
                        Threshold = Threshold - Delta;
                        visited_before = false;
                        return;
                    end
                else
                    Threshold = Threshold - Delta;
                    visited_before = false;
                    jj = j;
                    return;
                end
            end

        end

        function [P, C] = updateLLRsPSs(obj, i_start, i_cur, u_esti, P, C)
            n = log2(obj.code_length);
            if mod(i_cur, 2) ~= 0
                i_cur = i_cur - 1;
            end
            if mod(i_start, 2) ~= 0
                i_start = i_start - 1;
            end
            s_start = ffs(i_start, n);
            s_max = smax(i_start, i_cur, n);

            if s_start <= s_max
                i_minus1 = find_sMaxPos(s_start, s_max, i_start, n);
                for i = i_minus1:i_start
                    P = update_P(obj, i, P, C);
                    C = update_C(obj, i, C, u_esti(i+1));
                end
            else
                P = update_P(obj, i_start, P, C);
            end
        end

        function C = updatePSBack(obj, i_minus1, s_max, u_esti, C)
            k = 2^s_max;
            for i = i_minus1 + 1 - k:i_minus1
                C = update_C(obj, i, C, u_esti(i+1));
            end
        end

        function d = Viterbi_decoder(obj, llr, L)
            N = obj.code_length;
            g = obj.gen;
            m = obj.conv_depth - 1;
            obj.llr=llr;
            path_total = 2^m * L * 2;
            frozen_bits = ones(1, N);
            frozen_bits(obj.rate_profiling) = 0;
            P = zeros(N-1, path_total);
            C = zeros(N-1, 2*path_total);
            u_esti = zeros(N, path_total);
            v_esti = zeros(N, path_total);
            c_state = zeros(obj.conv_depth-1, path_total);
            M = zeros(1, path_total);
            active_path = zeros(1, path_total);
            active_path(1) = 1;
            for t = 0:N - 1
                active_path_idx = find(active_path==1);
                if frozen_bits(t+1) == 1
                    for path_index = active_path_idx
                        P(:, path_index) = update_P(obj, t, P(:, path_index), C(:, 2*path_index-1:2*path_index));
                        v_esti(t+1, path_index) = 0;
                        [u_esti(t+1, path_index), c_state(:, path_index)] = conv1bTrans(0, c_state(:, path_index), obj.gen);
                        M(path_index) = M(path_index) - m_func(P(1, path_index), u_esti(t+1, path_index));

                    end
                else
                    state_branch = cell(1,2^m);
                    for path_index = active_path_idx
                        P(:, path_index) = update_P(obj, t, P(:, path_index), C(:, 2*path_index-1:2*path_index));
                        %find available path
                        copy_path_index = 1;
                        while active_path(copy_path_index) == 1
                            copy_path_index=copy_path_index + 1;
                        end
                        active_path(copy_path_index) = 1;
                        %deepcopy
                        P(:, copy_path_index) = P(:, path_index);
                        C(:, 2*copy_path_index-1:2*copy_path_index) = C(:, 2*path_index-1:2*path_index);
                        c_state(:, copy_path_index) = c_state(:, path_index);
                        v_esti(:, copy_path_index) = v_esti(:, path_index);
                        u_esti(:, copy_path_index) = u_esti(:, path_index);
                        M(copy_path_index)= M(path_index);
                        %fork
                        v_esti(t+1, path_index) = 0;
                        v_esti(t+1, copy_path_index) = 1;
                        [u_esti(t+1, path_index), c_state(:, path_index)] = conv1bTrans(0, c_state(:, path_index), obj.gen);
                        [u_esti(t+1, copy_path_index), c_state(:, copy_path_index)] = conv1bTrans(1, c_state(:, copy_path_index), obj.gen);
                        M(copy_path_index) = M(copy_path_index) - m_func(P(1, copy_path_index), u_esti(t+1, copy_path_index));
                        M(path_index) = M(path_index) - m_func(P(1, path_index), u_esti(t+1, path_index));
                        s=0;
                        s_copy=0;
                        for i_s = 1:m
                        s = s + c_state(i_s, path_index)*2^(i_s-1);
                        s_copy = s_copy + c_state(i_s, copy_path_index)*2^(i_s-1);
                        end
                        s = s + 1;
                        s_copy = s_copy + 1;
                        state_branch{s}=[state_branch{s} path_index];
                        state_branch{s_copy}=[state_branch{s_copy} copy_path_index];
                    end

                    if sum(active_path) > path_total / 2
                        for branch_index = 1:2^m
                            num_branch = length(state_branch{branch_index});
                            if num_branch == 0
                                continue;
                            end
                            branch_path = state_branch{branch_index};
                            M_branch = M(branch_path);
                            [~,M_branch_sorted_indices] = sort(M_branch,'descend');
                            indices_kill = M_branch_sorted_indices(1:num_branch/2);
                            path_kill = branch_path(indices_kill);
                            active_path(path_kill) = 0;
                        end

                    end
                end
                active_path_idx = find(active_path==1);
                for path_index = active_path_idx

                    C(:, 2*path_index-1:2*path_index) = update_C(obj, t, C(:, 2*path_index-1:2*path_index), u_esti(t+1, path_index));
                end

            end

            active_path = logical(active_path);
            M_active = M(active_path);
            v_active = v_esti(:, active_path);
            [~, M_active_sorted] = sort(M_active, 'ascend');
            v = v_active(:, M_active_sorted(1));
            d = v(obj.rate_profiling);
        end

        function P = update_P(obj, phi, P, C)
            global operation_count_P;
            N = obj.code_length;
            n = log2(N);
            layer = obj.llr_layer_vec(phi + 1);

            switch phi %Decoding bits u_0 and u_N/2 needs channel LLR, so the decoding of them is separated from other bits.
                case 0
                    index_1 = obj.lambda_offset(n);
                    for beta = 0:index_1 - 1
                        P(beta + index_1) = sign(obj.llr(beta + 1)) * sign(obj.llr(beta + index_1 + 1)) * min(abs(obj.llr(beta + 1)), abs(obj.llr(beta + index_1 + 1)));
                        operation_count_P=operation_count_P+1;
                    end
                    for i_layer = n - 2:-1:0
                        index_1 = obj.lambda_offset(i_layer+1);
                        index_2 = obj.lambda_offset(i_layer+2);
                        for beta = 0:index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) * ...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)), abs(P(beta + index_1 + index_2)));
                            operation_count_P=operation_count_P+1;
                        end
                    end
                case N / 2
                    index_1 = obj.lambda_offset(n);
                    for beta = 0:index_1 - 1
                        x_tmp = C(beta + index_1, 1);
                        P(beta + index_1) = (1 - 2 * x_tmp) * obj.llr(beta + 1) + obj.llr(beta + 1 + index_1);
                        operation_count_P=operation_count_P+1;

                    end
                    for i_layer = n - 2:-1:0
                        index_1 = obj.lambda_offset(i_layer+1);
                        index_2 = obj.lambda_offset(i_layer+2);
                        for beta = 0:index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) * ...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)), abs(P(beta + index_1 + index_2)));
                            operation_count_P=operation_count_P+1;

                        end
                    end
                otherwise
                    index_1 = obj.lambda_offset(layer+1);
                    index_2 = obj.lambda_offset(layer+2);
                    for beta = 0:index_1 - 1
                        P(beta + index_1) = (1 - 2 * C(beta + index_1, 1)) * P(beta + index_2) + ...
                            P(beta + index_1 + index_2);
                        operation_count_P=operation_count_P+1;

                    end
                    for i_layer = layer - 1:-1:0
                        index_1 = obj.lambda_offset(i_layer+1);
                        index_2 = obj.lambda_offset(i_layer+2);
                        for beta = 0:index_1 - 1
                            P(beta + index_1) = sign(P(beta + index_2)) * ...
                                sign(P(beta + index_1 + index_2)) * min(abs(P(beta + index_2)), ...
                                abs(P(beta + index_1 + index_2)));
                            operation_count_P=operation_count_P+1;

                        end
                    end
            end
        end

        function C = update_C(obj, phi, C, u)
            global operation_count_C;
            N = obj.code_length;
            phi_mod_2 = mod(phi, 2);
            C(1, 1+phi_mod_2) = u;
            if (phi_mod_2 == 1) && (phi ~= N - 1)
                layer = obj.bit_layer_vec(phi + 1);
                for i_layer = 0:layer - 1
                    index_1 = obj.lambda_offset(i_layer+1);
                    index_2 = obj.lambda_offset(i_layer+2);
                    for beta = index_1:2 * index_1 - 1
                        C(beta + index_1, 2) = mod(C(beta, 1)+C(beta, 2), 2); %Left Column lazy copy
                        C(beta + index_2, 2) = C(beta, 2);
                        operation_count_C=operation_count_C+1;

                    end
                end
                index_1 = obj.lambda_offset(layer+1);
                index_2 = obj.lambda_offset(layer+2);
                for beta = index_1:2 * index_1 - 1
                    C(beta + index_1, 1) = mod(C(beta, 1)+C(beta, 2), 2); %Left Column lazy copy
                    C(beta + index_2, 1) = C(beta, 2);
                    operation_count_C=operation_count_C+1;

                end
            end
        end
    end
end
