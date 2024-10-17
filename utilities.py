'''
    Utility script 
    
    This script contains all functions necessary to run mainloop.py and simulation_OffAxis.py

    Python Version: 3.8.18
    numpy version: 1.24.3

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 17/10/2024
'''


# import libraries
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v2 as imageio 
import cv2 as cv

# Function to binarize an image
def binarize(im):
    """
    Convert an image to binary using a specified threshold.

    Parameters:
    im : The input image to binarize.

    Returns:
    im : The binarized image
    
    """
  
    im2=im.convert("L")
    threshold = 100
    im = im2.point(lambda p: p > threshold and 255)
    
    return im 


# Function to read an image 
def imageRead(namefile, size):
  """
    Read an image from a file, convert it to grayscale, and resize it.

    Parameters:
    namefile : Path to the image file (str).
    size : The new size for the image (width and height) (int).

    Returns: 
    loadImage : The resized grayscale image.
        
    """
    # inputs:
    # namefile - direction image to read
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    
    loadImage = loadImage.resize((size, size))

    return loadImage


# Function to display an Image
def imageShow(image, title):
    """
    Display an image with a specified title.

    Parameters:
    image : The image to display.
        
    title : The title for the displayed image (str).
        
    """
    plt.imshow(image, cmap='gray'), plt.title(title)
    plt.show()

    return


# Function to compute the Intensity of a given complex field
def intensity(complexField, log):
   """
    Compute the intensity of a given complex field.

    Parameters:
    - complexField : The input complex field.
    - log : If True, compute the log representation of the intensity (bool).
        
    Returns:
    - out : The computed intensity (and log if specified).
        
    """
    out = np.abs(complexField)
    out = out * out

    if log == True:
        out = 20 * np.log(out)
        out[out == np.inf] = 0
        out[out == -np.inf] = 0

    return out


# Function to compute the Fourier Transform
def ft(field):
    """
    Compute the Fourier Transform of a field.

    Parameters:
    - field : The input field for the Fourier Transform.
        
    Returns:
    - ft : The computed Fourier Transform.
        
    """
    ft = np.fft.fft2(field)
    ft = np.fft.fftshift(ft)

    return ft


# Function to compute the inverse Fourier Transform
def ift(field):
    """
    Compute the Inverse Fourier Transform of a field.

    Parameters:
    - field : The input field for the Inverse Fourier Transform.
        
    Returns:
    - ift : The computed Inverse Fourier Transform. numpy.ndarray    
    """
  
    ift = np.fft.ifft2(field)
    ift = np.fft.fftshift(ift)
    ift = np.fft.fftshift(ift)

    return ift


# Function to get image information
def imgInfo(img):
    """
    Get the dimensions of an image.

    Parameters:
    - img : The input image.
        
    Returns:
    - width, height : The dimensions of the image (width, height) (tuple)  
    """
    width, height = img.size
    #print(f"Image size: {width} x {height} pixels")
  
    return width, height


# Function to create a circular mask
def circularMask(width, height, radius, centX, centY, visualize):
    """
    Create a circular mask.

    Parameters:
    - width : size image Y.  
    - height : size image X. 
    - radius : Radius of the circular mask.
    - centX :  coordinate Y center
    - centY : coordinate X center
    - visualize :  If True, display the mask (bool).
       
    Returns:
    - mask : The created circular mask.
        
    """
    X, Y = np.ogrid[:width, :height]
    mask = np.zeros((width, height))
    circle = np.sqrt((X - centX) ** 2 + (Y - centY) ** 2) <= radius
    mask[circle] = 1

    if visualize:
        imageShow(mask, 'mask')

    return mask


# Function to save an Image
def saveImg(sample, name):
     """
    Save an image to a file after normalization.

    Parameters:
    - sample : The input image data.
    - name : The filename to save the image (str)
        
    """
    
    sample = np.abs(sample)
    
    image_data = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255)
    image = Image.fromarray(image_data.astype(np.uint8))
    imageio.imwrite(name, image)
    #image.save('name.png', format="png")

    return
