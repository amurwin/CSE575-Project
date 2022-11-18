from sklearn import cluster
from sklearn.cluster import KMeans

from functions import *
from KNN import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

#####################################################################
# PCA

n_components = 20

# Manual PCA
X_train_mean = X_train - np.mean(X_train)
covar = np.cov(X_train_mean, rowvar=False)
eigVal, eigVec = np.linalg.eigh(covar) # maybe use eigh instead rather than eig, because a covariance matrix is supposed to be symmetric
q = [(eigVal[i], eigVec[:,i]) for i in range(0, len(eigVal))]
q.sort(key = lambda x: x[0], reverse=True)
matrix = q[0][1]
for i in range (1, n_components):
    matrix = np.append(matrix, q[i][1])
matrix = matrix.reshape(n_components,-1)
X_train_pca = np.matmul(X_train, matrix.T)
X_test_pca = np.matmul(X_test, matrix.T)

explained_variance_ratio = [float(eigVal[i] / sum(eigVal)) for i in range(0, n_components)]
for i in range(n_components):
    print('Percentage of variance explained by PC {}: {}'.format(i+1, explained_variance_ratio[i]))
print('Total variance explained by 20 PCs: {}'.format(np.sum(explained_variance_ratio)))

pca_fig, pca_ax = plt.subplots(1)
pca_ax.plot(range(1, 21), [sum(explained_variance_ratio[0:x+1]) for x in range(0, len(explained_variance_ratio))])
pca_ax.set_xlabel('Num principle components')
pca_ax.set_ylabel('Fraction of Total Variance Explained')




# # Stats + graph
# for i in range(n_components):
#     print('Percentage of variance explained by PC {}: {}'.format(i+1, pca.explained_variance_ratio_[i]))
# print('Total variance explained by 20 PCs: {}'.format(np.sum(pca.explained_variance_ratio_)))

# fig, ax = plt.subplots(1)
# ax.plot(range(1, 21), [sum(pca.explained_variance_ratio_.tolist()[0:x+1]) for x in range(0, len(pca.explained_variance_ratio_))])
# ax.set_xlabel('Num principle components')
# ax.set_ylabel('Fraction of Total Variance Explained')
# print graph with `fig`

#####################################################################

#plot PCA components to see what K value to use

PCA_components = pd.DataFrame(X_train_pcaKmeans)
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans.py instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(PCA_components.iloc[:, :3])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

inertia_fig, intertia_ax = plt.subplots(1)
intertia_ax.plot(ks, inertias)
intertia_ax.set_xlabel('Number of Clusters, k')
intertia_ax.set_ylabel('Inertia, k')
intertia_ax.set_xticks(ks)
#plt.show() used to detect elbow point

#from kmeans, elbow point is roughly 5

#note: k_means might be useful for data visualization but thats about it

########################################################################
# KNN

#KNN via SKLEARN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_pca,y_train)
testResult = knn_classifier.predict(X_test_pca)
print("KNN Prediction",testResult)
print("Given Test Data",y_test.tolist())

# Manual KNN
DstType = 1
k = 5

X_full_pca = np.concatenate((X_train_pca, X_test_pca))
y_full = np.concatenate((y_train, y_test))

for i in range(91, len(y_full)):
    train_label = y_full.copy().tolist()
    del train_label[i]
    train_feat = X_full_pca.copy().tolist()
    del train_feat[i]

    pred_label = knn(X_full_pca[i], train_label, train_feat, k, DstType)
    print('Document ' + str(i) + ' groundtruth ' + str(y_full[i]) + ' predicted as ' + str(pred_label))

########################################################################
# Linear Regression

linReg(X_train_pca, y_train, X_test_pca, y_test)

os.chdir(owd)