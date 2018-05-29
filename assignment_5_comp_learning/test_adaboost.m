%% Test adaboost
clear all;
close all;
N_samples = 1000;
T_iterations = 100;

% Generate data
data = gendatb(N_samples);

features = data.data;
labels = str2num(data.labels) - 1; % PRtools has 1-indexed char labels

% Shuffle data
perm = randperm(N_samples);
features = features(perm, :);
labels = labels(perm, :);

% Split into training and testing data
features_train = features(1:round(N_samples / 2), :);
labels_train = labels(1:round(N_samples / 2));

features_test = features(round(N_samples / 2):end, :);
labels_test = labels(round(N_samples / 2):end);

% Train and test classifier
[betas, class_params] = adaboost(features_train, labels_train, T_iterations);
pred_labels = adaboost_classifier(features_test, betas, class_params);

error_rate = sum(pred_labels ~= labels_test) / length(labels_test);
fprintf('Error rate on test set: %.4f\n', error_rate);

% Illustrate
figure();
% Plot test set
gscatter(features_test(:,1), features_test(:,2), labels_test, ['b', 'm']);
hold on;
% Plot decision boundary
[xx, yy] = meshgrid(...
    (min(features_test(:,1)) - 1):.2:(max(features_test(:,1)) + 1),...
    (min(features_test(:,2)) - 1):.2:(max(features_test(:,2)) + 1));
ll = reshape(adaboost_classifier([xx(:), yy(:)], betas, class_params),...
    size(xx, 1), size(xx, 2));
contour(xx, yy, ll, 1, 'red');
hold off;
