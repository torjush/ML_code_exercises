clear all;
close all;

%% Test that high weight == high priority
features = [
    1.0, 0.4;
    1.0, 0.6;
    1.0, 0.8;
    1.0, 0.0
    ];

labels = [1, 0, 1, 0]';
weights = [1, 20, 1, 1]';

[f_star, theta_star, y_star] = train_extended_decision_stump(...
    features, labels, weights);

fprintf('\tf_star\ttheta_star\ty_star\nVal:\t%d\t%.4f\t\t%d\n',...
    f_star, theta_star, y_star);

pred_labels = decision_stump_classifier(f_star, theta_star, y_star, features);
assert(sum(pred_labels ~= [0 0 1 0]') == 0,...
    'Predicted labels not as expected when using weights');

fprintf('Predicted labels: [%d %d %d %d]\n', ...
    pred_labels(1), pred_labels(2), pred_labels(3), pred_labels(4));