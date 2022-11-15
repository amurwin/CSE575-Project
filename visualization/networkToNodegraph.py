import networkx as nx
from matplotlib import pyplot as plt
import os
import csv


# Function to sort the list of tuples by its second item
def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return (sorted(tup, key=lambda x: x[2]))

# file: filepath
# type 0-Spring, 1-Circular, 2-Both(default)
def graphNetwork(file,type=2):
    g = nx.Graph()
    filename = os.path.basename(file)[:-4]

    #read csv data
    data = []
    rowPos = 1
    colPos = 1
    maxDensity = -1

    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for col in row:
                # if edge exists
                if int(col) != 0:
                    data.append([rowPos,colPos,int(col)])
                    if int(col) > maxDensity:
                        maxDensity = int(col)
                colPos += 1

            rowPos += 1
            colPos = 1
    file.close()

    #set up nodes and edges
    for x in range(1,71):
        g.add_node(x)
    for item in Sort_Tuple(data):
        g.add_edge(item[0], item[1], weight=item[2])

    pos1 = nx.spring_layout(g, k=1, seed=0) #seed set to 0 for consistent generation
    pos2 = nx.circular_layout(g)
    #commented out code for bg color
    #ax = plt.axes()
    #ax.set_facecolor('black')

    #spring layout
    if type==1 or type==2:
        for edge in g.edges(data="weight"):
            nx.draw_networkx_edges(g, pos1, edgelist=[edge], alpha=(edge[2] / maxDensity), width=.5,edge_color='blue')

        nx.draw_networkx_nodes(g,pos1,node_size=5,node_color='black')
        nx.draw_networkx_labels(g,pos1,font_size=2, font_color='white')
        plt.title(filename)
        plt.savefig(f"./Graphs/{filename}_network_s.png", dpi=1000)
        plt.clf()

    #circular layout
    if type==2 or type==2:
        for edge in g.edges(data="weight"):
            nx.draw_networkx_edges(g, pos2, edgelist=[edge], alpha=(edge[2] / maxDensity), width=.5,edge_color='blue')

        nx.draw_networkx_nodes(g, pos2, node_size=5, node_color='black')
        nx.draw_networkx_labels(g, pos2, font_size=2, font_color='white')
        plt.title(filename)
        plt.savefig(f"./Graphs/{filename}_network_c.png", dpi=1000)
        plt.clf()
