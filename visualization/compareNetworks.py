import os
import compareMatrix as cm
import compareNodegraph as cg

path = './Comparisons'
if not os.path.exists(path):
    os.makedirs(path)

# get filepath
path = "./compareData"
dir_list = os.listdir(path)

# creates matrix visualization
# additional parameter; colored=0 for grayscale, colored=1 for color(default)
cm.compareNetworks(dir_list[0],dir_list[1])

# creates node graph visualization
cg.compareNetworks(dir_list[0],dir_list[1])
