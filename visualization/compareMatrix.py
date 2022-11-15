import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import csv

# file1/file2: filepath
# colored 0-black and white, 1-colored(default)
def compareNetworks(file1,file2,colored=1):
    filename1 = os.path.basename(file1)[:-4]
    filename2 = os.path.basename(file2)[:-4]
    data1 = []
    data2 = []

    # track for normalization
    maxDensity = -1

    # open csv files

    with open(file1, 'r') as file1:
        reader = csv.reader(file1)
        for row in reader:
            for col in row:
                data1.append(int(col))
                if int(col) > maxDensity:
                    maxDensity = int(col)
    file1.close()

    with open(file2, 'r') as file2:
        reader = csv.reader(file2)
        for row in reader:
            for col in row:
                data2.append(int(col))
                if int(col) > maxDensity:
                    maxDensity = int(col)
    file2.close()

    # data normalized
    density1 = [(x / maxDensity)*maxDensity for x in data1]
    density2 = [(x / maxDensity)*maxDensity for x in data2]

    # generate 70x70 data grid
    xplot=[]
    yplot=[]
    for num in range(1,71):
        for num2 in range(1,71):
            yplot.append(num)
            xplot.append(num2)

    # plot data
    fig, ax = plt.subplots()

    # color mapping
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['white','red','orange','yellow','green'],100)
    if colored==0:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['white','black'],100)

    # file1 data (outer square)
    im = ax.scatter(
        x=xplot,
        y=yplot,
        s=13,
        marker='s',  # use square as scatterplot marker
        edgecolors='none',  # remove rounded edges
        c=density1,
        cmap=cmap
    )

    # file2 data (inner square)
    ax.scatter(
        x=xplot,
        y=yplot,
        s=3,        # smaller size
        marker='s',
        edgecolors='none',
        c=density2,
        cmap=cmap
    )
    # remove whitespace on graph edge
    ax.set_xlim([0, 71])
    ax.set_ylim([0, 71])
    # equalize x axis and y axis scaling
    ax.set_aspect('equal', adjustable='box')

    #pyplot.show()
    plt.title(filename1+ "(outer) / "+filename2+"(inner)")
    plt.colorbar(im,cmap=cmap)
    fig = plt.figure(1)
    fig.savefig(f"./Comparisons/{filename1}-{filename2}_matrix.png", dpi=1000)
    fig.clear()
    plt.close(fig)
