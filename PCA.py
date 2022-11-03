from functions import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

owd = os.getcwd()
os.chdir('brainnetworks/CSVdata/')

data = []
order = []
for file in os.listdir():
    data.append(normalize(pd.read_csv(file, header=None).values).flatten())
    order.append(file[:9])

# GET LABELS
sex = getLabel(0, order)
math = getLabel(1, order)
creativity = getLabel(2, order)

X_train, X_test, y_train, y_test = train_test_split(data, math, test_size=0.2, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
n_components = 20

# Manual PCA
X_train_mean = X_train - np.mean(X_train)
covar = np.cov(X_train_mean, rowvar=False)
w, v = np.linalg.eig(covar)
q = [(w[i], v[:,i]) for i in range(0, len(w))]
q.sort(key = lambda x: x[0], reverse=True)
matrix = q[0][1]
for i in range (1, n_components):
    matrix = np.append(matrix, q[i][1])
matrix = matrix.reshape(n_components,-1)
X_train_pca = np.matmul(X_train, matrix.T)
X_test_pca = np.matmul(X_test, matrix.T)

# Built in PCA
pca = PCA(n_components=n_components, svd_solver='full').fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



# PCA STATS - Using built-in PCA, needs to be transfered over
for i in range(n_components):
    print('Percentage of variance explained by PC {}: {}'.format(i+1, pca.explained_variance_ratio_[i]))
print('Total variance explained by 20 PCs: {}'.format(np.sum(pca.explained_variance_ratio_)))

fig, ax = plt.subplots(1)
ax.plot(range(1, 21), [sum(pca.explained_variance_ratio_.tolist()[0:x+1]) for x in range(0, len(pca.explained_variance_ratio_))])
ax.set_xlabel('Num principle components')
ax.set_ylabel('Fraction of Total Variance Explained')
# print graph with `fig`


linReg(X_train_pca, y_train, X_test_pca, y_test)



os.chdir(owd)