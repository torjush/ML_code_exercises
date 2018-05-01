% Exercise: Online Gradient Descent (OGD)

clear all;
load coin_data;

a_init = [0.2, 0.2, 0.2, 0.2, 0.2]'; % initial action

n = 213; % is the number of days
d = 5; % number of coins

% we provide you with values R and G.
alpha = sqrt(max(sum(r.^2,2))); 
epsilon = min(min(r)); 
G = alpha/epsilon; 
R = 1; 

% set eta:
%%% your code here %%%
eta = R/(G*sqrt(n));

a = a_init; % initialize action. a is always a column vector

L = nan(n,1); % keep track of all incurred losses
A = nan(d,n); % keep track of all our previous actions

for t = 1:n
    
    % we play action a
    [l,g] = mix_loss(a,r(t,:)'); % incur loss l, compute gradient g
    
    A(:,t) = a; % store played action
    L(t) = l; % store incurred loss
    
    % update our action, make sure it is a column vector
    %%% your code here %%%
    a = a - eta * g;
    assert(size(a,2) == 1, 'a is not a column vector');
    % after the update, the action might not be anymore in the action
    % set A (for example, we may not have sum(a) = 1 anymore). 
    % therefore we should always project action back to the action set:
    a = project_to_simplex(a')';

end

% compute total loss
%%% your code here %%%
total_loss = sum(L);
fprintf('Total loss suffered by OGD strategy: %.4f\n', total_loss);
% compute total gain in wealth
%%% your code here %%%
gain = exp(-total_loss);
fprintf('Gain achieved with OGD strategy: %.4f\n', gain);
% compute best fixed strategy
% (you may make use of loss_fixed_action.m
% and optimization toolbox if needed)
%%% your code here %%%
% Make constraints
constr_A = -eye(d);
constr_B = zeros(d,1);

constr_Aeq = ones(1,d);
constr_beq = 1;

% Minimize and print results
[a_min, L_a_min] = fmincon(@loss_fixed_action, a_init,...
    constr_A, constr_B, constr_Aeq, constr_beq...
    );
fprintf('The optimal action was found to be a_min =');
fprintf('[%.4f, %.4f, %.4f, %.4f, %.4f]\n',...
    a_min(1), a_min(2), a_min(3), a_min(4), a_min(5)...
    );
fprintf('with a total loss of: %.4f\n', L_a_min);
% compute regret 
%%% your code here %%%
regret = total_loss - L_a_min;
fprintf('Regret using OGD strategy: %.4f (bound: %.4f)\n',...
    regret, R*G*sqrt(n)...
    );
%% plot of the strategy A and the coin data

% if you store the strategy in the matrix A (size d * n)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(A')
legend(symbols_str)
title('rebalancing strategy OGD')
xlabel('date')
ylabel('investment action a_t')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
