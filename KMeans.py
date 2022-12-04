import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
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
#Set random seed for reproducibility
np.random.seed(1000)

scaler = StandardScaler()
X = np.array(data)

n_components = 20

# Built in PCA for quick kmeans
pcaKMeans = PCA(n_components=n_components, svd_solver='full')

X_pca = pcaKMeans.fit_transform(X)
PCA_components = pd.DataFrame(X_pca)
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans.py instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(PCA_components.iloc[:, :3])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

plt.plot(ks, inertias)
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia, k')
plt.show()
# KMEANS
k_means = cluster.KMeans(n_clusters= 6)
k_means.fit(X_pca)
kmeansLabel = k_means.fit_predict(X_pca)
unique_labels = np.unique(kmeansLabel)
for i in unique_labels:
    plt.scatter(X_pca[kmeansLabel == i , 0] , X_pca[kmeansLabel == i , 1] , label = i)
plt.title("K-Means Clustering with 5 clusters and PCA")
plt.legend()
plt.show()