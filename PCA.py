from functions import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


os.chdir('brainnetworks/CSVdata/')

data = []
for file in os.listdir():
    data.append(normalize(pd.read_csv(file, header=None).values).flatten())


# We may want to implement a StandardScaler, but I skipped it for now

n_components = 20
pca = PCA(n_components=n_components).fit(data)


# PCA STATS
for i in range(n_components):
    print('Percentage of variance explained by PC {}: {}'.format(i+1, pca.explained_variance_ratio_[i]))
    
print('Total variance explained by 20 PCs: {}'.format(np.sum(pca.explained_variance_ratio_)))

fig, ax = plt.subplots(1)

ax.plot(range(1, 21), [sum(pca.explained_variance_ratio_.tolist()[0:x+1]) for x in range(0, len(pca.explained_variance_ratio_))])
ax.set_xlabel('Num principle components')
ax.set_ylabel('Fraction of Total Variance Explained')
plt.show()

# DATA PROCESSING CONTINUED
pca_data = pca.transform(data)