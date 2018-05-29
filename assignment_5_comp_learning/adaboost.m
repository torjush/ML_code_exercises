function [ betas, classifier_params ] = adaboost( features, labels, iterations )

N_samples = size(features, 1);
if nargin < 3
    iterations = 1000;
end%if

weights = ones(N_samples, 1) ./ N_samples;
betas = ones(iterations, 1);
for t = 1:iterations
    % Normalize weights
    weights = weights ./ sum(weights);
    
    % Train a weak learner
    [f, theta, y] = train_extended_decision_stump(...
        features, labels, weights);
    % Get labels
    pred_labels = decision_stump_classifier(f, theta, y, features);
    
    % Find and update betas and weights
    error = sum(weights .* (pred_labels ~= labels));
    beta = error / (1 - error);
    if beta == 0
        weights = ones(N_samples, 1) ./ N_samples;
    else
        weights = weights .* beta.^(1 - (pred_labels ~= labels));
    end%if
    betas(t) = beta;
    classifier_params(t) = struct('f', f, 'theta', theta, 'y', y);
end%for

end

