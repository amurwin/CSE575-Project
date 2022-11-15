import networkx as nx
from matplotlib import pyplot as plt
import os
import csv

def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return (sorted(tup, key=lambda x: x[2]))

def compareNetworks(file1,file2):
    g = nx.Graph()
    filename1 = os.path.basename(file1)[:-4]
    filename2 = os.path.basename(file2)[:-4]

    # read csv data
    file1Dat=[]
    # get raw data from file 1 for comparison
    with open(file1, 'r') as file1:
        reader = csv.reader(file1)
        for row in reader:
            for col in row:
                file1Dat.append(int(col))
    file1.close()

    data = []   # dataset for graphing
    pointer = 0   # tracks position for file1 comparison
    rowPos = 1  # tracks row position
    colPos = 1  # tracks column position
    maxDiff = -1  # tracks maximum diff for normalization

    # read file2 data and compare to file1 data
    with open(file2, 'r') as file2:
        reader = csv.reader(file2)
        for row in reader:
            for col in row:
                diff = file1Dat[pointer]-int(col)
                if diff != 0:
                    if diff > 0:
                        data.append([rowPos, colPos, abs(diff),'blue'])  # file1 has more density
                    else:
                        data.append([rowPos, colPos, abs(diff),'red'])  # file2 has more density
                    if diff > maxDiff:
                        maxDiff = diff
                colPos += 1
                pointer+=1

            rowPos += 1
            colPos = 1
    file2.close()

    # set up nodes and edges
    for x in range(1,71):
        g.add_node(x)
    for item in Sort_Tuple(data):
        g.add_edge(item[0], item[1], weight=item[2], side=item[3])

    # draw graph
    pos2 = nx.circular_layout(g)
    for edge in g.edges.data():
        nx.draw_networkx_edges(g, pos2, edgelist=[edge], alpha=(edge[2].get("weight") / maxDiff), width=.5,edge_color=edge[2].get("side"))

    nx.draw_networkx_nodes(g, pos2, node_size=5, node_color='black')
    nx.draw_networkx_labels(g, pos2, font_size=2, font_color='white')
    plt.title(filename1+ "(blue) / "+filename2+"(red)")
    plt.savefig(f"./Comparisons/{filename1}-{filename2}_graph.png", dpi=1000)
    plt.clf()