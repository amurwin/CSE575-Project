import os
os.chdir('brainnetworks/CSVdata/')

for file in os.listdir():
    lines = []
    with open(file,'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            f.write(line[:-2] + line[-1])