## Mamimum Flow 

This repository contains various implementations of capacity scaling algorithm anl linear programming for maximum flow problem as well as their comparisons on different configurations. Here are the list of the versions

**Regular Implementation:** This is the typical implementation of the capacity scaling algoritm as explain in the book Network Flows: Theory, Algorithms, and Applications by Orlin et. al. Yet, it has two options for path finding that are DFS and BFS.

**Heap Implementation:** This version uses a different scheme for updating ![Delta](https://latex.codecogs.com/gif.latex?%24%5CDelta%24). To do so, it starts with the same initial ![Delta](https://latex.codecogs.com/gif.latex?%24%5CDelta%24) and when no augmeting path can be found uses a heap that contains the edge capacities to determine the next ![Delta](https://latex.codecogs.com/gif.latex?%24%5CDelta%24) instead of dividing by 2 directly as in the original implementation.

**Linear Programming:** In this version, the problem is formulated as a linear program and resulting system is solved by scipy and gurobipy separately for comparison purposes.

To test the code:

* Clone the repo
* Install the requirements by pip or conda
* Run runner.py with the existing configuration to see a quick test ride.

You can also see the comparisons either by checking out the pdf file. If you wish, you can add more experiments by simply adding a json file in the same format with the existing ones and running it from the runner.
