import pandas as ps
import numpy as np  # rhymes with lumpy

# kNN Evaluation
# DstType, k self explanatory

def knn(test_feat, train_label, train_feat, k, DstType):

    # Calculate distance
    dist = []
    if(DstType == 1):
        for element in train_feat:
            dist.append(SumSquaredDistances(test_feat, element))
    elif(DstType == 2):
        for element in train_feat:
            dist.append(AngleBetweenVectors(test_feat, element))
    else:
        print("invalid input")

    # Find the top k nearest neighbors, and do the voting.
    posCount = 0
    negCount = 0

    pairedDist = [(dist[i], i) for i in range(0,len(dist))]
    pairedDist.sort(key=lambda x: x[0])
    relevantLabels = [train_label[x[1]] for x in pairedDist[0:k]]

    return sum(relevantLabels) / len(relevantLabels)

# kNN Distance Metrics
# DST type 1
def SumSquaredDistances(test_feat, train_feat):
    sum = 0
    for i in range(0, len(test_feat)):
        sum += (test_feat[i] - train_feat[i])**2
    return sum

# DST type 2
# Since we are doing the arccos: the smaller the value, the closer to the compared sample
def AngleBetweenVectors(test_feat, train_feat):
    unit_test_feat = test_feat / np.linalg.norm(test_feat)
    unit_train_feat = train_feat / np.linalg.norm(train_feat)
    return np.arccos(np.dot(unit_test_feat, unit_train_feat))