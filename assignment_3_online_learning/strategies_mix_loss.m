clear all;
close all;

mix_loss = @(p_t, z_t) -log(sum(p_t .* exp(-z_t)));

adv_moves = [0, 0, 1, 0; 0.1, 0, 0, 0.9; 0.2, 0.1, 0, 0];
T = length(adv_moves);
d = 3;

experts = eye(d);

% Strategy A - trust current most accurate expert
% Initialize variables
p_A = zeros(d, T);
p_A(:,1) = [1/3, 1/3, 1/3]';

losses_A = zeros(T, 1);
losses_A(1) = mix_loss(p_A(:,1), adv_moves(:,1));

expert_losses = zeros(d,T);
for i = 1:d
    expert_losses(i,1) = mix_loss(experts(:,i), adv_moves(:,1));
end

% Calculate decision for every time step
for t = 2:T
    exp_cum_loss = sum(expert_losses(:,1:t-1), 2);
    [~, i] = min(exp_cum_loss);
    p_A(i, t) = 1;
    losses_A(t) = mix_loss(p_A(:,t), adv_moves(:,t));
    for i = 1:d
        expert_losses(i,t) = mix_loss(experts(:,i), adv_moves(:,t));
    end
end
print_matrix(p_A, 't', 'p', 'Decisions from strategy A:');

% Strategy B - aggregate algorithm
% Initialize variables
eta = 1; % Agg = exp with eta == 1
weights = zeros(d, T);
weights(:,1) = [1, 1, 1]';

p_B = zeros(d, T);
p_B(:,1) = [1/3, 1/3, 1/3]';

losses_B = zeros(T,1);
losses_B(1) = mix_loss(p_B(:,1), adv_moves(:,1));

% Calculate decision for every time step
for t = 2:T
    weights(:, t) = weights(:, t-1) .* exp(-eta .* adv_moves(:,t-1));
    p_B(:, t) = weights(:, t) ./ sum(weights(:,t));
    losses_B(t) = mix_loss(p_B(:,t), adv_moves(:,1));
end

print_matrix(p_B, 't', 'p', 'Decisions from strategy B:');

% Calculate cumulative losses
cum_loss_A = sum(losses_A);
cum_loss_B = sum(losses_B);

% Find and compare to the best expert to get expert regret
best_expert = min(exp_cum_loss);
expert_regret_A = cum_loss_A - best_expert;
expert_regret_B = cum_loss_B - best_expert;

fprintf('Strategy\t|Total loss\t|Expert regret\n');
fprintf('A\t\t|%.4f\t\t|%.4f\n', cum_loss_A, expert_regret_A);
fprintf('B\t\t|%.4f\t\t|%.4f\n\n', cum_loss_B, expert_regret_B);

% Calculate the cumulative loss bound
cum_loss_bound = log(d) + best_expert;
fprintf('Upper bound for cumulative mix loss: %.4f\n', cum_loss_bound);