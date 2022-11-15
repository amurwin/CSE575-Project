import os
import csv

# helper tool to get highest density value from dataset
def maxdensity(file):
    maxDensity = -1

    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for col in row:
                if int(col) != 0:
                    if int(col) > maxDensity:
                        maxDensity = int(col)

    file.close()
    return maxDensity


# Get the list of all files and directories
# path = "./CSVdata"
path = os.path.dirname(os.getcwd()) + "/brainnetworks/CSVdata"

dir_list = os.listdir(path)
maxValue = -1
for file in dir_list:
    print("Scanning " + str(file) + "...")
    result = maxdensity(f"{path}/{file}")
    if result > maxValue:
        maxValue = result

print("max density in dataset is: " + str(maxValue))
