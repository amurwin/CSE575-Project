import os
import compareMatrix as cm
import compareNodegraph as cg

path = './Comparisons'
if not os.path.exists(path):
    os.makedirs(path)

# get filepath
path = "./compareData"
dir_list = os.listdir(path)

print("Running...")
# creates matrix visualization
# additional parameter; colored=0 for grayscale, colored=1 for color(default)
cm.compareNetworks(f"{path}/{dir_list[0]}",f"{path}/{dir_list[1]}")

# creates node graph visualization
cg.compareNetworks(f"{path}/{dir_list[0]}",f"{path}/{dir_list[1]}")
print("Comparison Generated...")
