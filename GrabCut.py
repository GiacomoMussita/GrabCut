from RGBImage import *
from GMM import GaussianMixture
import os
import networkx as nx
import numpy as np


class GrabCut:
    
    def __init__(self, image, rect_vertices, strip_width=1, K_GMM=5, gamma=50., lambda_cost=9., n_iter=5):
        """
        Initialization of the class that implements GrabCut algorithm

        Parameters
        ----------
        image : RGBImage object
            Image on which the algorithm performs the segmentation.
        rect_vertices : np.array of size (2, 2)
            List of the two vertices (x, y coordinates) given by the user
            defining the rectangular region of inference (box).
        strip_width : int, optional
            Number of pixel defining the thinkness of the box edges. 
            The default is 1.
        K_GMM : int, optional
            Number of components in each Gaussian mixture model.
            The default is 5 as suggested in Rother et. al 2004..
        gamma : float, optional
            Coeffient that weights the smoothness term in the cost function.
            The default is 50. as suggested in Rother et. al 2004..
        lambda_cost : float, optional
            Coefficient used to assign the capacity of edges between terminal
            vertices of the graph and hard constrained pixels.
            The default is 9..
        n_iter : int, optional
            Number of iterations of the algorithm. The default is 5.

        Returns
        -------
        None.

        """
        # 1D representation of the image
        self.image = image
        # 2D representation of the image
        self.pixel_grid = self.image.data.reshape(self.image.height, self.image.width, 3)
        
        self.rect_vertices = np.array(rect_vertices)
        self.strip_width = strip_width
        self.K_GMM = K_GMM
        self.gamma = gamma
        self.lambda_cost = lambda_cost
        self.n_iter = n_iter
        self.beta = 0
        
        # Initialize the trimap, the values for alpha and the background/foreground indeces
        self.Tb, self.Tu, self.Tf = self.initializeTrimap()
        self.index_b = None
        self.index_f = None
        self.alpha = np.empty(self.image.size)
        self.initializeAlphaValues()

        
        # Initialize vector K for the GMMs components
        self.comp_GMM = np.empty(self.image.size)
        
        # Initialize the values for the GMM of the foreground and the background.
        # Store the vector K
        self.GMM_b = None
        self.GMM_f = None
        self.initializeGMM()
        
        # Calculate beta coefficient
        self.calculateBeta()
        
        # Assign the smoothness term over the pixel of the image
        self.V_upright = np.zeros((self.image.height, self.image.width))
        self.V_right = np.zeros((self.image.height, self.image.width))
        self.V_downright = np.zeros((self.image.height, self.image.width))
        self.V_down = np.zeros((self.image.height, self.image.width))
        # Smoothness for hard constraints
        self.V_max = 0.
        self.calculateSmoothness()
        
        
        # Initialize graph
        self.graph = nx.Graph()
        self.initializeGraph()
        
        
    def initializeTrimap(self):
        """
        Initialize the trimap (Tb, Tu, Tf), with Tb region of true background
        (outside the box provided by the user), Tb undecided region with pixels
        within the box, and Tf region of true foreground (empty).

        Returns
        -------
        3 np.array of pixel's indices for Tb, Tu, Tf.

        """
        
        print('\nInitializing the trimap (Tb, Tu, Tf)...')
        Tf = [] # Indexes of foreground
        Tb = [] # Indexes of background
        Tu = [] # Indexes of remaining region
        
        # If the user has specify correctly the rectangle
        if self.rect_vertices.shape == (2, 2):
            # Coordinate of upper vertex
            xupper, yupper = self.rect_vertices[0]
            # Coordinate of lower vertex
            xlower, ylower = self.rect_vertices[1]
            if pixelIsInside(self.image, getImageOffset(self.image, xupper, yupper)) and\
                pixelIsInside(self.image, getImageOffset(self.image, xlower, ylower)):
                    
                for x in range(xupper-self.strip_width, xlower+self.strip_width+1, 1):
                    for y in range(yupper-self.strip_width, ylower+self.strip_width+1, 1):
                        # If the point belongs to the strip, it is included in Tb
                        if not (x > xupper and x < xlower and y > yupper and y < ylower):
                            if pixelIsInside(self.image, getImageOffset(self.image, x, y)):
                                Tb.append(getImageOffset(self.image, x, y))
                        # If the point does not belong to the strip, it is include in Tu
                        else:
                            if pixelIsInside(self.image, getImageOffset(self.image, x, y)):
                                Tu.append(getImageOffset(self.image, x, y))
                                
        return np.array(Tb), np.array(Tu), np.array(Tf)
    
    
    def initializeAlphaValues(self):
        """
        Assign alpha values to the user provided trimap. Pixels inside the 
        box are marked as foreground (alpha=1), the ones outside the box are
        marked as background (alpha=0)

        Returns
        -------
        None.

        """
        
        print('Initializing the opacity values (alpha)...')
        alpha_values = []
        index_b = []
        index_f = []
        for i in range(self.image.size):
            # If the pixel is within the undecided region, assign it to the foreground
            # (alpha = 1), otherwise to the background (alpha=2).
            if i in self.Tu:
                alpha_values.append(1)
                index_f.append(i)
            else:
                alpha_values.append(0)
                index_b.append(i)
                
        self.alpha = np.array(alpha_values)
        self.index_b = np.array(index_b)
        self.index_f = np.array(index_f)
        
        return
    
    
    def initializeGMM(self):
        """
        Initialization of the Gaussian Mixture models: for background it uses
        the Tb component of the trimap, for the foreground is uses Tu.
        Assignment of the vector K that keeps track of the most likely 
        component for each pixel of the image.

        Returns
        -------
        None.

        """
        
        
        print('Background and foreground GMMs initialization...')
        # Initialize the GMM and the vector K of the components
        self.GMM_b = GaussianMixture(self.image.data[self.index_b], self.K_GMM)
        self.comp_GMM[self.index_b] = self.GMM_b.K_init
        self.GMM_f = GaussianMixture(self.image.data[self.index_f], self.K_GMM)
        self.comp_GMM[self.index_f] = self.GMM_f.K_init
        
        return
    
    
    def calculateBeta(self):
        """
        Calculate beta coefficient necessary to assign the weights between
        neighbourhood pixels within the smoothness term of cost function.
        Formula for beta is provided in Rother et. al 2004.

        Returns
        -------
        None.

        """
        
        # Compute the mean of the difference between neighborhood pixels.
        # To consider each difference only once, each pixel is assumed to have
        # 4 neighborhood (but with 8-way connectivity):
        # upright, right, downright, down
        print('Calculating coefficient beta...')
        upright_diff = self.pixel_grid[1:, :-1] - self.pixel_grid[:-1, 1:]
        right_diff = self.pixel_grid[:, :-1] - self.pixel_grid[:, 1:]
        downright_diff = self.pixel_grid[:-1, :-1] - self.pixel_grid[1:, 1:]
        down_diff = self.pixel_grid[:-1, :] - self.pixel_grid[1:, :]
        
        # Sum of all the squared differences
        sum_diff = np.sum(np.square(upright_diff)) + np.sum(np.square(right_diff)) +\
            np.sum(np.square(downright_diff)) + np.sum(np.square(down_diff))
        # Total number of squared differences (divice by 3 to consider the colors)
        n_diff = (upright_diff.size + right_diff.size + downright_diff.size + down_diff.size)//3
        self.beta = 1 / (2 * sum_diff / n_diff)
        print('Beta = {:.2f}'.format(self.beta))
        
        return
    
    
    def calculateSmoothness(self):
        """
        Calculate the smoothness cost term V as in Rother et. al 2004.
        These costs are constant throughout the entire algorithm, therefore
        they are computed only once.
        Here is where the beta coefficient is used.

        Returns
        -------
        None.

        """
        
        # Calculate the smoothness term between neighborhood pixels.
        # This costs are fixed for all the iterations of the algorithm.
        print('Calculating smoothness cost function V...')
        for n in range(self.image.size):
            x, y = getImageCoordinate(self.image, n)
            
            # First row and last column pixels do no have upright neighbourhood.
            if y != 0 and x != (self.image.width - 1):
                deltaI = np.linalg.norm(self.pixel_grid[y, x] - self.pixel_grid[y-1, x+1])
                self.V_upright[y, x] = self.gamma*np.exp(-self.beta*deltaI)/np.sqrt(2)
                # print("V_upright:", self.V_upright[y, x])
                
            # Last column pixels do no have right neighbourhood.
            if x != (self.image.width - 1):
                deltaI = np.linalg.norm(self.pixel_grid[y, x] - self.pixel_grid[y, x+1])
                self.V_right[y, x] = self.gamma*np.exp(-self.beta*deltaI)
                
            # Last row and last column pixels do no have downright neighbourhood.
            if y != (self.image.height - 1) and x != (self.image.width - 1):
                deltaI = np.linalg.norm(self.pixel_grid[y, x] - self.pixel_grid[y+1, x+1])
                self.V_downright[y, x] = self.gamma*np.exp(-self.beta*deltaI)/np.sqrt(2)
                
            # Last row pixels do no have down neighbourhood.
            if y != (self.image.height - 1):
                deltaI = np.linalg.norm(self.pixel_grid[y, x] - self.pixel_grid[y+1, x])
                self.V_down[y, x] = self.gamma*np.exp(-self.beta*deltaI)
        
        # Set V_max really high so that the hard constrained pixels are not part of the cut
        self.V_max = self.lambda_cost*self.gamma
        

        return
    
    
    def assignGMMComponents(self):
        """
        Assign to each background pixel the most likely background GMM
        component.
        Assign to each foreground pixel the most likely foreground GMM
        component.

        Returns
        -------
        None.

        """
        
        # Assign each pixel in the undecided region Tu to a component of its GMM
        for index in self.Tu:
            
            # If the pixel is assigned to the background
            if self.alpha[index] == 0:
                self.comp_GMM[index] = self.GMM_b.assignComponent(self.image.data[index])
                
            # If the pixel is assigned to the foreground
            else:
                self.comp_GMM[index] = self.GMM_f.assignComponent(self.image.data[index])
        
        return
    
    
    def learnGMMParameters(self):
        """
        Update the mean, covariance and weights of background and foreground
        GMMs based on the current alpha values and on the current most
        likely components (K vector Rother et al. 2004).

        Returns
        -------
        None.

        """
        
        # Update the parameters of the GMM
        self.GMM_b.learnParameters(self.image.data[self.index_b], self.comp_GMM[self.index_b])
        self.GMM_f.learnParameters(self.image.data[self.index_f], self.comp_GMM[self.index_f])
        
        return
    
    
    def initializeGraph(self):
        """
        Build the graph:
            - two terminals vertices 'S' (source) representing the foreground,
            and 'T' (sink) representing the background
            - one node for each pixel
            - edges between neighbourhood pixels weighted based on the 
            smoothness function
            

        Returns
        -------
        None.

        """
        
        print('Graph initialization...')
        # Add source and sink nodes
        self.graph.add_nodes_from(('S', 'T'))
        for n in range(self.image.size):
            # Add all the pixel as nodes
            self.graph.add_node(n)
            # Add edges from source ('S', foreground) and sink ('T', background) to all pixels
            self.graph.add_edges_from([('S', n), ('T', n)])
        # Add edges between adjacent pixels
        for n in range(self.image.size):
            x, y = getImageCoordinate(self.image, n)
            
            if y != 0 and x != (self.image.width - 1):
                self.graph.add_edge(n, getImageOffset(self.image, x+1, y-1),
                                    capacity=self.V_upright[y, x])
                
            if x != (self.image.width - 1):
                self.graph.add_edge(n, getImageOffset(self.image, x+1, y),
                                    capacity=self.V_right[y, x])

                
            if y != (self.image.height - 1) and x != (self.image.width - 1):
                self.graph.add_edge(n, getImageOffset(self.image, x+1, y+1),
                                    capacity=self.V_downright[y, x])

                
            if y != (self.image.height - 1):
                self.graph.add_edge(n, getImageOffset(self.image, x, y+1),
                                    capacity=self.V_down[y, x])
                
        return
    
    
    def updateDataCost(self):
        """
        Update the weights of the edges between pixels' nodes and terminal
        vertics. These are weighted as in Boykov et al. 2001.
        The edges between 'S' and pixels have a weights as minus the 
        loglikelihood of the pixel color to belong to the background
        color model. 
        The edges between 'T' and pixels have a weights asminus the 
        loglikelihood of the pixel color to belong to the foreground
        color model.
        To ensure the respect of hard constraints, user specified
        background pixels should be assigned to 'T': therefore the weights
        of the edges between this pixels and 'T' is high, so that these
        edges are not included in the cut. The value is defined by variable
        self.V_max (which depend on parameter lambda_cost).

        Returns
        -------
        None.

        """
        
        # Update the weights of the terminal nodes costs
        for n in range(self.image.size):
            # If the pixel belongs to an undecided region
            if n in self.Tu:
                # Data term for background and foreground
                D_n_b = -self.GMM_b.returnLogProb(self.image.data[n])
                self.graph['S'][n]['capacity'] = D_n_b
                D_n_f = -self.GMM_f.returnLogProb(self.image.data[n])
                self.graph['T'][n]['capacity'] = D_n_f
            # If the pixel is an hard constrained pixel:
            # if it belongs to the background
            elif n in self.index_b:
                self.graph['T'][n]['capacity'] = self.V_max
                self.graph['S'][n]['capacity'] = 0
            # if it belongs to the foreground (this is never run actually) 
            else:
                self.graph['T'][n]['capacity'] = 0
                self.graph['S'][n]['capacity'] = self.V_max
                
        return
 
    
    def estimateSegmentation(self):
        """
        Update the graph by calling updateDataCost and computing the cut by
        calling the networkx library function nx.minimum_cut().
        Update the alpha values and the indices of background and foreground.

        Returns
        -------
        None.

        """
        
        
        # Estimate the segmentation using min-cut algorithm
        
        # Update the data cost U (weights of the edge from sink and source)
        print("- updating the graph with the new terminal costs...")
        self.updateDataCost()
        # Compute the cut
        print("- performing the min-cut...")
        cost, cut = nx.minimum_cut(self.graph, 'S', 'T', capacity='capacity')
        cut[0].remove('S')
        cut[1].remove('T')
        
        print('Cut complete:')
        print('Nodes assigned to foreground = ', len(cut[0]))
        print('Nodes assigned to background = ', len(cut[1]))
        print('Minimum cost (value of the cut) = {:.2f}'.format(cost))
        
        # Update alpha values for background and foreground:
        
        # if the pixel is assigned to the source 'S' (foreground)
        for node in cut[0]:
                self.alpha[node] = 1
                
        # if the pixel is assigned to the sink 'T' (background)
        for node in cut[1]:
                self.alpha[node] = 0
        
        # Update the indeces of background and foreground pixels.
        self.index_b = [n for n in range(self.image.size) if self.alpha[n] == 0]
        self.index_f = [n for n in range(self.image.size) if self.alpha[n] == 1]
        
        return
    
    
    def runAlgorithm(self):
        """
        Run the main algorithm:
            1. assign GMMs components
            2. learn GMMs parameters
            3. estimate segmentation
            4. repeat from step 1 (for the defined number of iterations)

        Returns
        -------
        None.

        """
        
        for iteration in range(self.n_iter):
            print("\nIteration: ", iteration+1)
            # At the first iteration, updating the parameters of the GMM is not necessary, 
            # since it has been already done in the initialization of the GMMs.
            if iteration != 0:
                print("1. Assigning GMMs components...")
                self.assignGMMComponents()
                print("2. Learning GMMs parameters...")
                self.learnGMMParameters()
            print("3. Estimating the segmentation using min-cut:")
            self.estimateSegmentation()
            self.saveSegmentationImage('output', 'segmentation' + str(iteration+1) + '.png')
        
        return
    
    
    def saveRectImage(self, path=None, file_name='rect.png'):
        """
        Helper function to save the image with the user box.

        Parameters
        ----------
        path : str, optional
            Path in which save the image. The default is None.
        file_name : str, optional
            Name of the image. The default is 'rect.png'.

        Returns
        -------
        None.

        """
    
        if path is not None:
            if os.path.exists(path):
                rect = np.copy(self.image.data)
                red = np.array([1., 0., 0.])
                for i in self.Tb:
                    rect[i] = red
                # Reshape image
                rect = rect.reshape(self.image.height, self.image.width, 3)
                # Convert the image to RGB values (from 0 to 255)
                rect = rect * 255
                rect = Image.fromarray(rect.astype(np.uint8), mode="RGB")
                rect.save(os.path.join(path, file_name))
                # rect.show()

        return
    
    
    def saveTrimapImage(self, path=None, file_name='trimap.png'):
        """
        Helper function to save the image with the user trimap.

        Parameters
        ----------
        path : str, optional
            Path in which save the image. The default is None.
        file_name : str, optional
            Name of the image. The default is 'trimap.png'.

        Returns
        -------
        None.

        """
        
        if path is not None:
            if os.path.exists(path):
                trimap = np.copy(self.image.data)
                red = np.array([1., 0., 0.])
                white = np.array([1., 1., 1.])
                for i in range(self.image.size):
                    if self.alpha[i] == 0:
                        trimap[i] = white
                for i in self.Tb:
                    trimap[i] = red
                # Reshape image
                trimap = trimap.reshape(self.image.height, self.image.width, 3)
                # Convert the image to RGB values (from 0 to 255)
                trimap = trimap * 255
                trimap = Image.fromarray(trimap.astype(np.uint8), mode="RGB")
                trimap.save(os.path.join(path, file_name))
                # rect.show()
        
        return
    
    
    def saveSegmentationImage(self, path=None, file_name='segmentation.png'):
        """
        Helper function to save the image of the segmentation.

        Parameters
        ----------
        path : str, optional
            Path in which save the image. The default is None.
        file_name : str, optional
            Name of the image. The default is 'segmentation.png'.

        Returns
        -------
        None.

        """
        
        if path is not None:
            if os.path.exists(path):
                segmentation = np.copy(self.image.data)
                white = np.array([1., 1., 1.])
                for i in range(self.image.size):
                    if self.alpha[i] == 0:
                        segmentation[i] = white
                # Reshape image
                segmentation = segmentation.reshape(self.image.height, self.image.width, 3)
                # Convert the image to RGB values (from 0 to 255)
                segmentation = segmentation * 255
                segmentation = Image.fromarray(segmentation.astype(np.uint8), mode="RGB")
                segmentation.save(os.path.join(path, file_name))
        
        return
    
    
    def saveImageMatting(self, path=None, file_name='imagematting.png'):
        """
        Helper function to save the image with the matting.

        Parameters
        ----------
        path : str, optional
            Path in which save the image. The default is None.
        file_name : str, optional
            Name of the image. The default is 'imagematting.png'.

        Returns
        -------
        None.

        """
        
        if path is not None:
            if os.path.exists(path):
                matting = np.copy(self.image.data)
                black = np.array([0., 0., 0.])
                white = np.array([1., 1., 1.])
                for i in range(self.image.size):
                    if self.alpha[i] == 0:
                        matting[i] = black
                    else:
                        matting[i] = white
                # Reshape image
                matting = matting.reshape(self.image.height, self.image.width, 3)
                # Convert the image to RGB values (from 0 to 255)
                matting = matting * 255
                matting = Image.fromarray(matting.astype(np.uint8), mode="RGB")
                matting.save(os.path.join(path, file_name))
        
        return
        