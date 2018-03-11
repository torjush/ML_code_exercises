# # ## Semi-supervised 2
# # Train a k-nearest-neighbors classifier and predict labels for the unlabeled data, then train LDA on this data

# from sklearn.neighbors import KNeighborsClassifier

# k_nearest = KNeighborsClassifier(weights='distance', n_neighbors=3)
# k_nearest.fit(X_l, y_l)

# # Predict labels for unlabeled data
# y_u_pred = k_nearest.predict(X_u)
# print("Accuracy on prediction for \"unlabeled\" data: {:.4f}". format(accuracy_score(y_pred=y_u_pred, y_true=y_u)))

# # Fit classifier with true labels where availible, predicted otherwise
# clf = LinearDiscriminantAnalysis()
# clf.fit(X_l, y_l)
# clf.fit(X_u, y_u_pred)

# y_tot_pred = clf.predict(X_tot)
# print("Accuracy on prediction for labeled and \"unlabeled\" data: {:.4f}". format(accuracy_score(y_pred=y_tot_pred, y_true=y_tot)))


# # ## Semi-supervised 2
# # Train a k-means cluster, and classify each unlabeled point in the cluster from the labeled points in the same cluster

# from sklearn.cluster import KMeans

# cluster = KMeans(n_clusters=2)
# cluster.fit(X_tot)

# # Assign labels from clustering
# zero_in_zero = 0
# zero_in_one = 0

# for i in range(len(y_l)):
#     if y_l[i] == cluster.labels_[i] and y_l[i] == 0:
#         zero_in_zero += 1
#     elif y_l[i] == cluster.labels_[i] and y_l[i] == 1:
#         zero_in_one += 1
            
# if zero_in_zero > zero_in_one:
#     y_u_kmeans = cluster.labels_[len(y_l):]
# else:
#     y_u_kmeans = np.asarray([0 if label == 1 else 1 for label in cluster.labels_[len(y_l):]])

# y_u_kmeans.shape

# # Combine labeled and data with cluster-labels to train LDA
# X_clustered = np.concatenate((X_l, X_u), axis=0)
# y_clustered = np.concatenate((y_l, y_u_kmeans), axis=0)

# y_clustered.shape

# clf = LinearDiscriminantAnalysis()
# clf.fit(X_tot, y_clustered)

# y_tot_pred = clf.predict(X_tot)
# print("Accuracy on prediction for labeled and \"unlabeled\" data: {:.4f}". format(accuracy_score(y_pred=y_tot_pred, y_true=y_tot)))

