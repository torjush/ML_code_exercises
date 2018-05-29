clear all;
close all;

% Constants
N_samples = 10000;

mu_1 = [0;0];
mu_2 = [2;0];

sigma = eye(2);

% Generate data from both distributions
features_1 = mvnrnd(mu_1, sigma, N_samples / 2);
features_2 = mvnrnd(mu_2, sigma, N_samples / 2);

labels_1 = zeros(N_samples / 2, 1);
labels_2 = ones(N_samples / 2, 1);

% Combine
features = cat(1, features_1, features_2);
labels = cat(1, labels_1, labels_2);

% Shuffle
perm = randperm(size(features, 1));
features = features(perm, :);
labels = labels(perm, :);

% Train the decision stump
[f_star, theta_star, y_star] = train_decision_stump(features, labels);

fprintf('\tf_star\ttheta_star\ty_star\nVal:\t%d\t%.4f\t\t%d\n',...
    f_star, theta_star, y_star);