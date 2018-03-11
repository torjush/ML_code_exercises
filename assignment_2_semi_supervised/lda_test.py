import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X = np.random.randn(9,9)
y = np.random.randint(0,1, 9)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
