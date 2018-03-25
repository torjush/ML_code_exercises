function [loss, grad] = loss_fixed_action(a_fixed)
    % [loss, grad] = LOSS_FIXED_ACTION(a_fixed)
    % given a fixed action a_fixed, compute the total loss of that action
    % for the complete bitcoin dataset
    %
    % Input:
    % a_fixed = the fixed action (column vector)
    %
    % Outputs:
    % loss = total loss of action a_fixed
    % grad = total gradient of the loss (column vector)
    
    load coin_data;
    n = 213;

    for t = 1:n
        [l,g] = mix_loss(a_fixed,r(t,:)'); % incur loss l, compute gradient g
        L(t) = l;
        G(:,t) = g;
    end
    
    loss = sum(L);
    grad = sum(G,2);

end