from RGBImage import *
from GrabCut import *
import os
import numpy as np

""" Uncomment the parameters of the desired image """


""" Parameters for sunflower_sky.png: the image has few colors (blue for the background,
and yellow and green for the foreground), therefore it is better to have few components
for the color model."""

# Dimentions are reduced for computational reasons
img_width, img_height = 525, 350

# Positions of the upper left vertex and lower right vertex of the box
upper_left_vertex = [140, 90]
lower_right_vertex = [380, 348]
rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

strip_width = 1 # thickness of the user defined box
K_GMM = 2 # number of components for the Gaussian mixture model
gamma = 50 # weight assigned to the smoothness in the cost function
lambda_cost = 9 # cost of the edges for hard constrained pixels
n_iter = 3 # number of iterations
input_path = os.path.join('data', 'sunflower_sky.png')
output_path = 'output'


""" Parameters for duck.png"""

# # Dimentions are reduced for computational reasons
# img_width, img_height  = 256, 256 

# # Positions of the upper left vertex and lower right vertex of the box
# upper_left_vertex = [70, 25]
# lower_right_vertex = [200, 200]
# rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

# strip_width = 1
# K_GMM = 2
# gamma = 50
# lambda_cost = 9
# n_iter = 2
# input_path = os.path.join('data', 'duck.png')
# output_path = 'output'


""" Parameters for messi.png"""

# # Dimentions are reduced for computational reasons
# img_width, img_height  = 400, 267 

# # Positions of the upper left vertex and lower right vertex of the box
# upper_left_vertex = [100, 5]
# lower_right_vertex = [300, 260]
# rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

# strip_width = 1
# K_GMM = 5
# gamma = 50
# lambda_cost = 9
# n_iter = 3
# input_path = os.path.join('data', 'messi.png')
# output_path = 'output'


""" Parameters for ball.png"""

# # Dimentions are reduced for computational reasons
# img_width, img_height  = 377, 265 

# # Positions of the upper left vertex and lower right vertex of the box
# upper_left_vertex = [30, 30]
# lower_right_vertex = [210, 200]
# rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

# strip_width = 1
# K_GMM = 5
# gamma = 50
# lambda_cost = 9
# n_iter = 3
# input_path = os.path.join('data', 'ball.png')
# output_path = 'output'


""" Parameters for child.png"""

# # Dimentions are reduced for computational reasons
# img_width, img_height  = 512, 512

# # Positions of the upper left vertex and lower right vertex of the box
# upper_left_vertex = [180, 40]
# lower_right_vertex = [440, 490]
# rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

# strip_width = 1
# K_GMM = 5
# gamma = 50
# lambda_cost = 9
# n_iter = 4
# input_path = os.path.join('data', 'child.png')
# output_path = 'output'


""" Parameters for tennis.png"""

# # Dimentions are reduced for computational reasons
# img_width, img_height  = 427, 240

# # Positions of the upper left vertex and lower right vertex of the box
# upper_left_vertex = [60, 40]
# lower_right_vertex = [190, 190]
# rect_vertices = np.array([upper_left_vertex, lower_right_vertex])

# strip_width = 1
# K_GMM = 7
# gamma = 50
# lambda_cost = 9
# n_iter = 3
# input_path = os.path.join('data', 'tennis.png')
# output_path = 'output'


img = RGBImage(input_path, img_height, img_width)
grabcut = GrabCut(img, rect_vertices, strip_width, K_GMM, gamma, lambda_cost, n_iter)
grabcut.saveRectImage(output_path)
grabcut.saveTrimapImage(output_path)
grabcut.runAlgorithm()
grabcut.saveSegmentationImage(output_path)
grabcut.saveImageMatting(output_path)






