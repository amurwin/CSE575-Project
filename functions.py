import numpy as np

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