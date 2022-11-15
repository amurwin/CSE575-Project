import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import csv

# file: filepath
# colored 0-black and white, 1-colored(default)
def graphNetwork(file, colored=1):
    #read csv
    filename = os.path.basename(file)[:-4]
    data = []
    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            rowArray=[]
            for col in row:
                rowArray.append(int(col))
            data.append(rowArray)
    file.close()

    #determine colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['white','red','orange','yellow','green'],100)
    if colored==0:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['white','black'],100)

    #plot data
    img = plt.imshow(data,interpolation='nearest',cmap = cmap,origin='lower')

    #save and adjust graph
    plt.colorbar(img,cmap=cmap)
    plt.title(filename)
    fig = plt.figure(1)
    fig.savefig(f"./Graphs/{filename}_matrix.png")
    fig.clear()
    plt.close(fig)