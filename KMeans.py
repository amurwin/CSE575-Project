import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from functions import normalize, getLabel

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

n_components = 20

# Built in PCA for quick kmeans
pcaKMeans = PCA(n_components=n_components, svd_solver='full').fit(X_train)
X_train_pcaKmeans = pcaKMeans.transform(X_train)
X_test_pcaKmeans = pcaKMeans.transform(X_test)

# KMEANS
k_means = cluster.KMeans(n_clusters= 5)
k_means.fit(X_train_pcaKmeans)
kmeansLabel = k_means.fit_predict(X_test_pcaKmeans)
unique_labels = np.unique(kmeansLabel)
for i in unique_labels:
    plt.scatter(X_test_pcaKmeans[kmeansLabel == i , 0] , X_test_pcaKmeans[kmeansLabel == i , 1] , label = i)
plt.title("K-Means Clustering with 5 clusters and PCA")
plt.legend()
plt.show()