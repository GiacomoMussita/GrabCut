# GrabCut
Python implementation of GrabCut algorithm
<br />Author: Giacomo Mussita
<br />Method: GrabCut
<br />Paper: Rother et al. 2004, “GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts
<br />Language: Python
<br />Dependency: numpy, networkx, sklearn


### FILE DESCRIPTION:
- main.py: main function for parameters configuration and starts the algorithm.
- GrabCut.py: core implementation of the algorithm (class 'GrabCut').
- GMM.py: implementation of the Gaussian Mixture Model (class 'GMM').
- RGBImage.py: helper script for importing images.
- folder \data contains the input images (place here new image), folder \output is for the outputs.


### ALGORITHM:
0. Initialization of the trimap, of the opacity values and of the GMMs models.
1. Assign GMM components to pixels in the region of inference
2. Learn GMM parameters from the image
3. Estimate the segmentation for the given models (Graph Cut to minimize the
cost function E = U + V, U data/color term, V smoothness term)
4. Repeat from step 1 for a fixed number of iteration
5. Apply border matting


### USAGE:
- open the code in main.py.
- uncomment the desired image definition (or change the name of the image in 'output_path').
- modify the algorithm parameters if desired.
- for new images, specify the positions of upper-left and lower-right pixels of the rectangle containing the object (variable 'rect_vertices'). 
- run the script (it will requires approximately 4min for 3 iterations on a 525x350 image, reduce iteration to 2 for lower execution time).
- results are saved in folder \output.


### RELEVANT POINTERS:
- line 220, GrabCut.py -> method 'calculateSmoothness' computes the smoothness term V of the cost function.

- line 313, GrabCut.py -> method 'initializeGraph' build the structure of the graph assigning the neighbourhood pixels' weights based on V.

- line 164, GMM.py -> method 'assignComponent' performs step 1 of the algorithm. For each component in the GMM, it calls method 'calculateCostComponent' (line 127, GMM.py) to estimate the probability.

- line 71, GMM.py -> method 'learnParameters' implements step 2 of the algorithm.

- line 362, GrabCut.py -> method 'updateDataCost' modifies the graph at each ieration by updating the terminal-pixel edges based on the new GMM. The costs assigned reflects the data/color term U of the cost function. This method implements a part of step 3.


### NOT IMPLEMENTED:
- Kmeans clustering for GMM initialization. I used sklearn for that.
- Minimum cut on graph algorithm used in Rother et al. 2004. I used 'minimum_cut' function of library networkx, which uses a different (slower) implementation.
- Border matting.
