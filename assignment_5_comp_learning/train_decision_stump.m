function [ f_star, theta_star, y_star ] = train_decision_stump( features, labels)
N_features = size(features, 2);
M_samples = size(features, 1);

assert(size(labels, 1) == M_samples);
feature_maxes = max(features, [], 1);
feature_mins =  min(features, [], 1);
feature_ranges = feature_maxes - feature_mins;

predicted_labels = zeros(M_samples, 1);
best_error = M_samples;% Initialize to worst case, everything wrong
for f = 1:N_features
    for threshold = feature_mins(f):feature_ranges(f)/100:feature_maxes(f)
        for sign = 1:2
            predicted_labels = decision_stump_classifier(...
                f, threshold, sign, features);
            error = sum(predicted_labels ~= labels);
            if error < best_error
                f_star = f;
                theta_star = threshold;
                y_star = sign;
                best_error = error;
            end%if
        end%for
    end%for
end%for

end

