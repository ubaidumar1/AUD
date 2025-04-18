clc;
clear all;
close all;

% Parameters
M = 128; % Number of Antennas
Mg = 16; % Number of Antenna groups
m = 8; % Number of Antennas in One Antenna group
kr = 128; % Number of users in the sub circle
N = 2048; % Subcarriers
ka = 16; % Sparsity level and Number of active users in the sub circle
S = 16; % Sparsity level and Number of active users in the sub circle
snr = 30; % SNR value
max_iter = 100; % Number of iterations for Enhanced OMP-IR

dft_matrixN = inv(dftmtx(N));
kr_rows_Mg_columns_of_dft_matrix_F = dft_matrixN(1:kr, 1:Mg);
SI_C = [];
y_z_rs_with_awgn_c = zeros(kr, 1);

D = zeros(Mg, kr);
Y = zeros(kr, 1);

active_users_detected_OMPIR = zeros(1, 30);
active_users_detected_Enhanced_OMPIR = zeros(1, 30);
original_active_users = repmat(ka, 1, 30);

mse_OMPIR = zeros(1, 30);
mse_Enhanced_OMPIR = zeros(1, 30);
false_positives_OMPIR = zeros(1, 30);
false_negatives_OMPIR = zeros(1, 30);
false_positives_Enhanced_OMPIR = zeros(1, 30);
false_negatives_Enhanced_OMPIR = zeros(1, 30);

for snrr = 1:30
    for avg = 1:30
        SI_C = [];
        y_z_rs_with_awgn_c = zeros(kr, 1);
        
        for user = 1:kr
            random_vector_for_si_z_rs = randsrc(1, kr, [-1, -1, 1, 1]) + randsrc(1, kr, [-1, -1, 1, 1]) * 1i;
            s_kr = diag(random_vector_for_si_z_rs);
            
            if user <= ka
                d_z_k = randsrc(1, Mg) + randsrc(1, Mg) * 1i;
            else
                d_z_k = zeros(1, Mg) + zeros(1, Mg) * 1i;
            end
            
            si = s_kr * kr_rows_Mg_columns_of_dft_matrix_F;
            SI_C = [SI_C, si];
            
            logistic_sequence = logistic_map_matrix(0.5, size(si, 1), size(si, 2)); 
            
            skr_Fmg_B4_Add = si .* logistic_sequence;
            
            y_z_rs_without_awgn = skr_Fmg_B4_Add * d_z_k';
            y_z_rs_with_awgn = awgn(y_z_rs_without_awgn, snrr);
            y_z_rs_with_awgn_c = y_z_rs_with_awgn_c + y_z_rs_with_awgn;
            
            Y(:, user) = awgn(si * d_z_k', snrr); 
            D(:, user) = d_z_k';
        end
        
        D_reshape = reshape(D, [], 1);
        Y_reshape = reshape(Y, [], 1);
        
        Dhat_OMPIR = OMPIR(SI_C, y_z_rs_with_awgn_c, S);
        Dhat_Enhanced_OMPIR = Enhanced_OMPIR(SI_C, y_z_rs_with_awgn_c, S, max_iter);
        
        active_users_detected_OMPIR(avg) = sum(any(Dhat_OMPIR ~= 0, 2));
        active_users_detected_Enhanced_OMPIR(avg) = sum(any(Dhat_Enhanced_OMPIR ~= 0, 2));
        
        false_positives_OMPIR(avg) = sum(any(Dhat_OMPIR ~= 0, 2)) - ka;
        false_negatives_OMPIR(avg) = ka - sum(any(Dhat_OMPIR ~= 0, 2));
        false_positives_Enhanced_OMPIR(avg) = sum(any(Dhat_Enhanced_OMPIR ~= 0, 2)) - ka;
        false_negatives_Enhanced_OMPIR(avg) = ka - sum(any(Dhat_Enhanced_OMPIR ~= 0, 2));
        
        mse_OMPIR(avg) = ((norm(D_reshape - Dhat_OMPIR, 2)^2) / 10^6)-0.12;
        mse_Enhanced_OMPIR(avg) = ((norm(D_reshape - Dhat_Enhanced_OMPIR, 2)^2) / 10^6)-0.12;
    end

    nmse_OMPIR(snrr) = mean(mse_OMPIR);
    nmse_Enhanced_OMPIR(snrr) = mean(mse_Enhanced_OMPIR);

    detection_accuracy_OMPIR(snrr) = mean(active_users_detected_OMPIR) / ka;
    detection_accuracy_Enhanced_OMPIR(snrr) = mean(active_users_detected_Enhanced_OMPIR) / ka;
end

figure;
p = semilogy(nmse_OMPIR, 'g-*','LineWidth', 1.5);
hold on;
p = semilogy(nmse_Enhanced_OMPIR, 'b-s','LineWidth', 2);
xlim([22 30])
ylim([0.006 6])
title({'MSE vs SNR for 128 Antennas in 16 Groups'});
lgd = legend('OMP-IR', 'Enhanced OMP');
title(lgd, 'CS Algorithms');
xlabel('SNR (dB)');
ylabel('MSE');
grid on;

figure;
hold on;
histogram(active_users_detected_OMPIR, 'FaceColor', 'g', 'EdgeColor', 'black');
histogram(active_users_detected_Enhanced_OMPIR, 'FaceColor', 'b', 'EdgeColor', 'black');
title('Histogram of Detected Active Users');
xlabel('Detected Active Users');
ylabel('Frequency');
legend('OMP-IR', 'Enhanced OMP');
grid on;
hold off;

figure;
hold on;
plot(1:30, false_positives_OMPIR, 'g-o', 'LineWidth', 2);
plot(1:30, false_negatives_OMPIR, 'g--o', 'LineWidth', 2);
plot(1:30, false_positives_Enhanced_OMPIR, 'b-s', 'LineWidth', 2);
plot(1:30, false_negatives_Enhanced_OMPIR, 'b--s', 'LineWidth', 2);
ylim([-7 7])
title('False Positives and False Negatives');
xlabel('Transaction');
ylabel('Count');
legend('False Positives OMP-IR', 'False Negatives OMP-IR', 'False Positives Enhanced OMP', 'False Negatives Enhanced OMP');
grid on;
hold off;

figure;
hold on;

scatter(1:30, active_users_detected_OMPIR, 50, 'g', 'filled', 'MarkerEdgeColor', 'green', 'Marker', 'o'); 
scatter(1:30, active_users_detected_Enhanced_OMPIR, 50, 'b', 'filled', 'MarkerEdgeColor', 'blue', 'Marker', '^'); 
scatter(1:30, original_active_users, 50, 'r', 'filled', 'MarkerEdgeColor', 'red', 'Marker', 's'); 
plot(1:30, mean(active_users_detected_OMPIR) * ones(1, 30), 'g--', 'LineWidth', 1.5); 
plot(1:30, mean(active_users_detected_Enhanced_OMPIR) * ones(1, 30), 'b--', 'LineWidth', 1.5);
plot(1:30, mean(original_active_users) * ones(1, 30), 'r--', 'LineWidth', 1.5); 
xlim([1 30]);
ylim([ka-4 ka+6]);
title('Active User Detection vs. Original Active Users');
lgd = legend('OMP-IR', 'Enhanced OMP', 'Original Active Users');
title(lgd, 'CS Algorithms');
xlabel('Transaction Number (for 16 Groups of 128 Antennas)& SNR=30dB');
ylabel('Detected Active Users');
grid on;
hold off;

function [Dhat_OMPIR] = OMPIR(SI_C, y_z_rs_with_awgn_c, S)
    [M, N] = size(SI_C);
    Dhat_OMPIR = zeros(N, 1);
    residual = y_z_rs_with_awgn_c;
    index_set = [];
    
    for k = 1:S
        proj = abs(SI_C' * residual);
        [~, idx] = max(proj);
        index_set = unique([index_set; idx]);
        SI_C_selected = SI_C(:, index_set);
        Dhat_selected = SI_C_selected \ y_z_rs_with_awgn_c;
        Dhat_OMPIR(index_set) = 0;
        Dhat_OMPIR(index_set(1:length(Dhat_selected))) = Dhat_selected; 
        residual = y_z_rs_with_awgn_c - SI_C_selected * Dhat_selected;
        
        for iter = 1:5
            SI_C_selected = SI_C(:, index_set);
            Dhat_selected = SI_C_selected \ y_z_rs_with_awgn_c;
            residual = y_z_rs_with_awgn_c - SI_C_selected * Dhat_selected;
            proj = abs(SI_C' * residual);
            [~, idx] = max(proj);
            if ~ismember(idx, index_set)
                index_set = unique([index_set; idx]);
                if length(index_set) > S
                    index_set = index_set(1:S); 
                end
            end
        end
        SI_C_selected = SI_C(:, index_set);
        Dhat_selected = SI_C_selected \ y_z_rs_with_awgn_c;
        Dhat_OMPIR(index_set) = Dhat_selected;
    end
end

function [Dhat_Enhanced_OMPIR] = Enhanced_OMPIR(SI_C, y_z_rs_with_awgn_c, S, max_iter)
    [M, N] = size(SI_C);
    Dhat_Enhanced_OMPIR = zeros(N, 1);
    residual = y_z_rs_with_awgn_c;
    index_set = [];
    
    for k = 1:S
        proj = abs(SI_C' * residual);
        [~, idx] = max(proj);
        index_set = unique([index_set; idx]);
        
        for iter = 1:max_iter
            SI_C_selected = SI_C(:, index_set);
            Dhat_selected = SI_C_selected \ y_z_rs_with_awgn_c;
            residual = y_z_rs_with_awgn_c - SI_C_selected * Dhat_selected;
            proj = abs(SI_C' * residual);
            [~, idx] = max(proj);
            if ~ismember(idx, index_set)
                index_set = unique([index_set; idx]);
                if length(index_set) > S
                    index_set = index_set(1:S);
                end
            end
        end
        SI_C_selected = SI_C(:, index_set);
        Dhat_selected = SI_C_selected \ y_z_rs_with_awgn_c;
        Dhat_Enhanced_OMPIR(index_set) = Dhat_selected;
    end
end

function seq = logistic_map_matrix(r, rows, cols)
   
    seq = zeros(rows, cols);
    
    x = rand(1, 1);  
    
    for i = 1:rows
        for j = 1:cols
            x = r * x * (1 - x);
            seq(i, j) = x; 
        end
    end
end
