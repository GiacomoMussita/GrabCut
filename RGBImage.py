import numpy as np
import os
from PIL import Image
                
class RGBImage:
    
    def __init__(self, path=None, img_height=None, img_width=None):
        """
        Transform RGB images into numpy array

        Parameters
        ----------
        path : str, optional
            Path of the image.
        img_height : str, optional
            Desired height of the image. Pass a value if you want to resize the image.
        img_width : str, optional
            Desired width of the image. Pass a value if you want to resize the image.

        Returns
        -------
        None.

        """
        self.data = []
        self.width = 0
        self.height = 0
        
        if path is not None:
            if os.path.exists(path):
                img = Image.open(path)
                if img_height is not None and img_width is not None:
                    img = img.resize((img_width, img_height))
                # img.show()
                # Transform the image into a numpy array
                img = np.asarray(img)
                # Normalize the RBG values between 0 and 1
                img = img / 255
                # Dimentions of the image
                width = img.shape[1]
                height = img.shape[0]
                # Flatten the image
                img = np.reshape(img, (img.shape[0]*img.shape[1], 3))
                self.data = img
                self.width = width
                self.height = height
                self.size = width*height

            
def getImageOffset(image, x, y):
    """
    Returns the index of the image as stored in memory (1D numpy array) given the coordinates (x, y). 
    The function checks if the pixel is within the image.
    If the pixel is outside the image, the function returns -1.

    Parameters
    ----------
    image : RGBImage object
        Source image.
    x : int
    y : int

    Returns
    -------
    offset : int
        Index of the pixel in the 1D image.

    """
    offset = -1
    if (x >= 0 and x < image.width and y >= 0 and y < image.height):
        offset = (y * image.width) + x
    return offset


def getImageCoordinate(image, offset):
    """
    Returns the coordinates (x, y) of the pixel in the image, give the index of the image.
    The function checks if the pixel is within the image.
    If the pixel is outside the image, the function returns -1 both for x and y.
    
    Parameters
    ----------
    image : RGBImage object
        Source image.
    offset : int
        Index of the pixel in the 1D image.

    Returns
    -------
    x : int
    y : int

    """
    
    x = -1
    y = -1
    if (offset >= 0 and offset < image.size):
        x = offset % image.width
        y = offset // image.width 
    return x, y


def pixelIsInside(image, pixel):
    """
    Check if a pixel is inside an image. Return either True of False.

    Parameters
    ----------
    image : RGBImage object
        Souce image.
    pixel : int
        Index of the pixel in the 1D image to be checked.

    Returns
    -------
    bool
        True if the pixel is inside the image, False if it is outside.

    """
    
    if pixel >= 0 and pixel < image.size:
        return True
