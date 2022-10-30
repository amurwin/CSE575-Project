import functions
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


os.chdir('brainnetworks/CSVdata/')

data = []
order = []
for file in os.listdir():
    data.append(functions.normalize(pd.read_csv(file, header=None).values).flatten())
    order.append(file[:9])


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
# plt.show()

# DATA PROCESSING CONTINUED
pca_data = pca.transform(data)

# GET LABELS
# sex = functions.getLabel(0, order)
math = functions.getLabel(1, order)
# creativity = functions. getLabel(2, order)

# SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(data, math, test_size=0.2, random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

functions.linReg(X_train, y_train, X_test, y_test)
