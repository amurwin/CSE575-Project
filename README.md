# CSE575 Brain Network Analysis

Derek Deng, Ben Hay, Vincent Le, Drew Murwin Breanna Seitz 

We want to classify individual brain networks into three different categories: sex, high-math capability, and creativity. There are two options for classifying each network. One option is to classify each category individually, where three separate machine learning models will interpret each brain network for each of the categories. This method will require the development of three separate models. The second option is more complex, where a single machine learning model will interpret classifications for all three categories. This method encounters the multiple comparisons problem (MCP), which occurs when multiple hypotheses are tested simultaneously. A solution to this problem has been noted in a report that has done similar work on brain networks [2]. This report also has results for identifying sex via brain networks, giving us existing grounds to work on. The data collected contains brain networks along with associated data such as sex, age, and other metrics [3]. The interpretation of high-math capability and creativity will be based on these collected metrics. Due to the higher-dimensional nature of data, the curse of dimensionality and developing useful visualizations of results also need to be addressed. This research will further increase the understanding of brain connectivities. In addition, it could potentially be used to differentiate between mental illnesses and atypical math/social capabilities, and lead to further advancements that could increase human brain capabilities.

In order to Visualize the data you must run visualizeNetworks.py. If you want to specifically compare two datasets, you must copy the two .csv files from the /CSVdata folder into the /compareData folder and run compareNetworks.py which will produce a matrices graph and a node-link graph.

References:

[1] Alper, Basak, et al. “Weighted Graph Comparison Techniques for Brain Connectivity 
Analysis.” www.microsoft.com, 27 Apr. 2013, 
research.microsoft.com/en-us/um/people/nath/docs/brainvis_chi2013.pdf. Accessed 15 
Sept. 2022.

[2] Duarte-Carvajalino, Julio M., et al. “Hierarchical Topological Network Analysis of 
Anatomical Human Brain Connectivity and Differences Related to Sex and Kinship.” 
NeuroImage, vol. 59, no. 4, 15 Feb. 2012, pp. 3784–3804, 
www.sciencedirect.com/science/article/pii/S1053811911012687, 
10.1016/j.neuroimage.2011.10.096. Accessed 15 Sept. 2022.

[3] “Index of /User/Lakoglu/Courses/95828/S17/Projectsources/brainnetworks.rar.” 
Www.andrew.cmu.edu, 
www.andrew.cmu.edu/user/lakoglu/courses/95828/S17/projectsources/brainnetworks.rar. 
Accessed 15 Sept. 2022.
