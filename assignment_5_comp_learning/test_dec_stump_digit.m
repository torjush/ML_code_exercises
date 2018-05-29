clear all;
close all;

N_train = 50;

% Prepare data with labels
data = table2array(importfile('optdigitsubset.txt'));
zero_data = data(1:554,:);
one_data = data(555:end,:);

zero_labels = zeros(size(zero_data, 1), 1);
one_labels = ones(size(one_data, 1), 1);

features = cat(1, zero_data, one_data);
labels = cat(1, zero_labels, one_labels);

% Shuffle it
perm = randperm(size(features, 1));
features = features(perm, :);
labels = labels(perm, :);

% Pick out training and testing sets
features_train = features(1:N_train, :);
labels_train = labels(1:N_train, :);


features_test = features(N_train:end, :);
labels_test = labels(N_train:end, :);

% Train the decision stump
[f_star, theta_star, y_star] = train_decision_stump(...
    features_train, labels_train);

fprintf('Parameters found from training on %d samples:\n', N_train);
fprintf('\tf_star\ttheta_star\ty_star\nVal:\t%d\t%.4f\t\t%d\n',...
    f_star, theta_star, y_star);

pred_labels_test = decision_stump_classifier(...
    f_star, theta_star, y_star, features_test);
error_rate = sum(double(labels_test ~= pred_labels_test)) / size(features_test, 1);
fprintf('Error rate: %.4f\n', error_rate);