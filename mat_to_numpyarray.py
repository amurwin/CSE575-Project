import scipy.io
mat = scipy.io.loadmat('brainnetworks/smallgraphs/M87102217_fiber.mat')
array = mat['fibergraph'].toarray()
print(array)
