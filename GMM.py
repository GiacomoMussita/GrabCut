import numpy as np
from sklearn.cluster import KMeans

seed = 4365

class GaussianMixture:
    
    def __init__(self, image, n_components=5):
        """
        Initialization of the class that implements the Gaussian mixture model.

        Parameters
        ----------
        image : np.array of size (number of pixels, 3)
            Array containing the values of each pixel of the RGB image.
        n_components : int, optional
            Number of components for the GMM. The default is 5.

        Returns
        -------
        None.

        """
        
        self.n_components = n_components
        # The number of features corresponds to the number of colors
        self.n_features = image.shape[1]
        # Array to store the means, the weights and the covarice matrices
        # of the components
        self.means = np.zeros((self.n_components, self.n_features))
        self.pi_weights = np.zeros(self.n_components)
        self.sigma = np.zeros((self.n_components, self.n_features, self.n_features))
        
        # To keep track of the inverse and of the determinant (to speed up the computations):
        # sigma_inv store the intermediate results for the matrix inverse for each
        # component, sigma_det stores the intermediate results for the determinant
        # of the covariance matrices.
        self.sigma_inv = np.zeros((self.n_components, self.n_features, self.n_features))
        self.sigma_det = np.zeros(self.n_components)
        
        # Initialize the first clusters
        self.K_init = self.initialize(image)
        
        
    def initialize(self, image):
        """
        To build a Gaussian mixture model.
        First, Kmeans clustering is applied (from sklearn), from the cluster,
        the parameters of model are initialized by method learnParameters

        Parameters
        ----------
        image : np.array of size (number of pixels, 3)
            Array containing the values of each pixel of the RGB image.

        Returns
        -------
        Kn : list
            Ordered list of GMM components assigned to each pixel.

        """
        
        # Build n_components clusters for the pixel in data array X.
        # Assign each pixel to the correct cluster.
        # A seed is assigned to ensure repeatability.
        Kn = KMeans(n_clusters=self.n_components, n_init=1, random_state=seed).fit(image).labels_
        self.learnParameters(image, Kn)
        return Kn
    
    
    def learnParameters(self, image, Kn):
        """
        Function that updated the means, the weights and the covariance 
        matrices of each component as in Rother et al. 2004.
        Update also the partial results for covariance inverse matrices and
        covariance matrices determinants.

        Parameters
        ----------
        image : np.array of size (number of pixels, 3)
            Array containing the values of each pixel of the RGB image.
        Kn : np.array of size (number of pixels)
            Ordered list of GMM components assigned to each pixel.

        Returns
        -------
        None.

        """
        
        # Re-initialize the values of parameters
        self.means = np.zeros((self.n_components, self.n_features))
        self.pi_weights = np.zeros(self.n_components)
        self.sigma = np.zeros((self.n_components, self.n_features, self.n_features))
        
        # Compute the model parameters for each component
        for k in range(self.n_components):
            pixels_in_k = []
            # Build a list of pixels belonging the k-th component
            for n in range(image.shape[0]):
                if Kn[n] == k:
                    pixels_in_k.append(n)
            # If at least one pixel is assigned to the k-th component,
            # it is possible to compute the parameters
            if len(pixels_in_k) > 0:
                self.pi_weights[k] = len(pixels_in_k)
                self.means[k] = np.mean(image[pixels_in_k], axis=0)
                self.sigma[k] = np.zeros((self.n_features, self.n_features)) if len(pixels_in_k) <= 1 else np.cov(image[pixels_in_k].T)
                # Note: numpy cannot calculate the covariance of a single vector.
                # This happen when only one pixel is assigned to the component.
                
                # It is necessary that the determinant of the matrix is different from zero,
                # otherwise it causes problem with the logarithm and the inverse
                self.sigma_det[k] = np.linalg.det(self.sigma[k])
                noise = 0.01
                if self.sigma_det[k] == 0.:
                    self.sigma[k] += noise*np.eye(self.n_features)
                    self.sigma_det[k] = np.linalg.det(self.sigma[k])
                self.sigma_inv[k] = np.linalg.inv(self.sigma[k])
        # Normalized the values of pi_weights
        if np.sum(self.pi_weights) > 0:
            self.pi_weights = self.pi_weights / np.sum(self.pi_weights)
              
        return
    
    
    def calculateCostComponent(self, pixel, k):
        """
        Calculate the probability of the pixel to belong to a given component.
        The formula is a simple Gaussian as in Rother et al. 2004:
            prob(z|alpha, k, GMM) = pi(k)*exp(-0.5(z - mu(k))*(sigma(k))^-1*(z - mu(k))^T)/sqrt(det(sigma(k)))

        Parameters
        ----------
        pixel : np.array of size (1, 3)
            RGB values of a pixel.
        k : int
            Number of the component to considered.

        Returns
        -------
        prob : float
            Probability of the pixel to belong to the component.

        """
        
        # Calculate the weighted probability of the pixel to belong to the component.
        
        # The weight of the component must be greater than zero, otherwise the
        # probability is zero.
        # Actually a small noise is added to the probability to avoid computing
        # the log-probability of zero.
        
        prob = 0.001
        
        if self.pi_weights[k] > 0.:
            
            prob = (self.pi_weights[k] / np.sqrt(np.abs(self.sigma_det[k]))) *\
                np.exp(-0.5*(pixel - self.means[k]) @ self.sigma_inv[k][:][:] @ (pixel - self.means[k]).T)

        return prob
    
    
    def assignComponent(self, pixel):
        """
        Find the most likely GMM component given the pixel .

        Parameters
        ----------
        pixel : np.array of size (1, 3)
            RGB values of a pixel.

        Returns
        -------
        kn : int
            Label of the most likely component.

        """
        
        # Assign the pixel to the most likely component of the GMM
        # Calculate the probability for each component
        prob = [self.calculateCostComponent(pixel, k) for k in range(self.n_components)]

        kn = np.argmax(np.array(prob))
        
        return kn
    
    def returnLogProb(self, pixel):
        """
        Compute the log-likelihood of the pixel to belong to the GMM.
        The likelihood is the weighted sum of the probability of belonging
        to each one of the components.

        Parameters
        ----------
        pixel : np.array of size (1, 3)
            RGB values of a pixel.

        Returns
        -------
        log_prob : float
            Log-likelihood.

        """
        
        # Return the log-probability for the pixel calcuated on the GMM model
        prob_component = np.array([self.calculateCostComponent(pixel, k) for k in range(self.n_components)])
        # To avoid applying the logarithm of zero a very large negative number is assigned
        if np.sum(prob_component) == 0.:
            log_prob = - 10e10
        else:
            log_prob = np.log(np.sum(prob_component))
        
        return log_prob
