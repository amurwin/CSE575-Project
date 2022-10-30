import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def normalize(array):
  # w_ij = a_ij / sum_j (a_ij)

  # create 2D array of 0s same size as array
  w = [[0.0 for j in range(array.shape[0])] for i in range(array.shape[1])]
  w = np.array(w)

  # iterate through rows
  for i in range(len(array)):
    # sum_j (a_ij)
    s = sum(array[i])
    if s != 0:
      # iterate through each value
      for j in range(len(array[i])):
        # w_ij = a_ij / sum_j (a_ij)
        w[i][j] = array[i][j] / s
  return w

def getLabel(label, order):
  # 0 = sex
  # 1 = math
  # 2 = creativity

  mapping = {0: 'Sex', 1: 'FSIQ', 2: 'CCI'}

  labels = pd.read_csv('../metainfo.csv')

  #reorder to match data
  labels = labels.set_index('URSI')
  labels = labels.reindex(index=order)
  labels.reset_index()

  return labels[mapping[label]]



def linReg(X_train, y_train, X_test, y_test):

  LR = LinearRegression()
  LR = LR.fit(X_train, y_train)
  W = LR.coef_
  b = LR.intercept_
  print("W: ", W)
  print("b: ", b)
  y_predict = LR.predict(X_test)
  error = np.sqrt(mean_squared_error(y_test, y_predict))
  print("Root mean squared error: ", error)
