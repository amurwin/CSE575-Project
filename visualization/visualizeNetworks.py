import os
import networkToNodegraph as ntg
import networkToMatrix as ntm

# file will output all visualizations into "Graphs" folder in this directory
path = './Graphs'
if not os.path.exists(path):
    os.makedirs(path)

# Get the list of all files and directories
#restore below line to use "CSVdata" in directory
#path = "./CSVdata"
path = os.path.dirname(os.getcwd()) + "/brainnetworks/CSVdata"

dir_list = os.listdir(path)

for file in dir_list:
    print("Visualizing "+str(file)+"...")

    # creates matrix visualization
    # additional parameter; colored=0 for grayscale, colored=1 for color(default)
    ntm.graphNetwork(f"{path}/{file}")
    # creates node graph visualization
    # additional parameter; type=0 for spring layout, type=1 for circular, type=2 for both(default)
    ntg.graphNetwork(f"{path}/CSVdata/{file}")
