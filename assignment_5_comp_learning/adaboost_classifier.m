function [ pred_labels ] = adaboost_classifier( features, betas, classifiers )
N_samples = size(features, 1);
T_weak_learners = size(betas, 1);
pred_labels = zeros(N_samples, 1);

threshold = 1/2 * sum(log(1./betas));

for i = 1:N_samples
    hyp_value = 0;
    for t = 1:T_weak_learners
        hyp_value = hyp_value + log(1 / betas(t)) * decision_stump_classifier(...
            classifiers(t).f, classifiers(t).theta,...
            classifiers(t).y, features(i,:));
    end%for
    if hyp_value >= threshold
        pred_labels(i) = 1;
    else
        pred_labels(i) = 0;
    end
end%for


end

