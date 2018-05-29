function [ pred_labels ] = decision_stump_classifier( f, theta, y, features )
N_samples = size(features, 1);

if y == 1
    pred_labels = double(features(:, f) > theta);
else
    pred_labels = double(features(:, f) < theta);
end%if
end

