% Exercise: Aggregating Algorithm (AA)
clear all;
load coin_data;

d = 5;
n = 213;

% compute adversary movez z_t
%%% your code here %%%
z = -log(r);

% compute strategy p_t (see slides)
%%% your code here %%%
p = zeros(n, d);
p(1,:) = ones(1,d) .* (1/d);

weights = zeros(n, d);
weights(1,:) = ones(1,d);

for t = 2:n
    weights(t,:) = weights(t-1,:) .* exp(-z(t-1,:));
    p(t,:) = weights(t,:) ./ sum(weights(t,:)); 
end

% compute loss of strategy p_t
%%% your code here %%%
losses = -log(sum(p.*exp(-z), 2));

% compute losses of experts
%%% your code here %%%
experts = eye(d);
expert_losses = zeros(n,d);
for i = 1:n
    for j = 1:d
        expert_losses(i,j) = -log(sum(experts(j,:).*exp(-z(i,:))));
    end
end

total_expert_losses = sum(expert_losses, 1);
fprintf('Expert\t|Total Loss\t|\n');
for i = 1:d
    fprintf('%s\t|\t%.4f\t|\n', symbols_str{i}, total_expert_losses(i));
end

% compute regret
%%% your code here %%%
regret = sum(losses) - min(total_expert_losses);
fprintf(...
    'Expert regret of aggregating algorithm: %.4f (bound: %.4f)\n',...
    regret, log(d)...
);

% compute total gain of investing with strategy p_t
%%% your code here %%%
gain = 1;
for i = 1:n
    gain = gain * (p(i,:) * r(i,:)');
end

fprintf('Total gain of aggregating algorithm: %.4f\n', gain);
%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
