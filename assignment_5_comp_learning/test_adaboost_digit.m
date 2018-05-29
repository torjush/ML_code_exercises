clear all;
close all;

train_size = 50;
T_iterations = 1000;

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

features_train = features(1:train_size, :);
labels_train = labels(1:train_size, :);

features_test = features(train_size + 1:end, :);
labels_test = labels(train_size + 1:end, :);
test_size = size(labels, 1) - train_size;

[betas, class_params] = adaboost(features_train, labels_train,...
    T_iterations);

pred_labels = adaboost_classifier(features_test, betas, class_params);

error_rate = sum(pred_labels ~= labels_test) / test_size;
fprintf('Error rate when trained on %d digits: %.4f\n', train_size, error_rate);