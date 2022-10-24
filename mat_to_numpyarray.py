import pandas as pd
import numpy as np
import scipy.io


# Convert .mat file to array
mat = scipy.io.loadmat('brainnetworks/smallgraphs/M87102217_fiber.mat')
array = mat['fibergraph'].toarray()
# print(array)

array2 = pd.read_csv('brannetworks/CSVdata/M87102217_fiber.csv').values
