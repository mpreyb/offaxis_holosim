# -*- coding: utf-8 -*-
"""
Title:-->            utility script
Description:-->
Authors:-->          Maria Paula Rey, and Raul Castaneda
Emaila:-->           mpreyb@eafit.edu.co
                     racastaneq@eafit.edu.co
Date:-->             04/08/2023
Date last modified:  03/26/2024
University-->        EAFIT University (Applied Optics Group)

Python Version: 3.8.18
numpy version: 1.24.3
"""

# import lybraries
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v2 as imageio 
import cv2 as cv

def binarize(im):
    im2=im.convert("L")
    #im2.save("b.jpg")
    threshold = 100
    im = im2.point(lambda p: p > threshold and 255)
    
    return im 


def imageRead(namefile, size):
    # inputs:
    # namefile - direction image to read
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    
    loadImage = loadImage.resize((size, size))

    return loadImage


# Function to display an Image
def imageShow(image, title):
    # inputs:
    # image - The image to show
    # title - Title of the displayed image
    plt.imshow(image, cmap='gray'), plt.title(title)
    plt.show()

    return


# Function to compute the amplitude of a given complex field
def amplitude(complexField, log):
    # inputs:
    # complexField - The input complex field to compute the amplitude
    # log - boolean variable to determine if a log representation is applied
    out = np.abs(complexField)

    if log == True:
        out = 20 * np.log(out)

    return out


# Function to compute the Intensity of a given complex field
def intensity(complexField, log):
    # inputs:
    # complexField - The input complex field to compute the intensity
    # log - boolean variable to determine if a log representation is applied
    out = np.abs(complexField)
    out = out * out

    if log == True:
        out = 20 * np.log(out)
        out[out == np.inf] = 0
        out[out == -np.inf] = 0

    return out


# Function to compute the phase of a given complex field
def phase(complexField):
    # inputs:
    # complexField - The input complex field to compute the phase
    out = np.angle(complexField)

    return out


# Function to compute the Fourier Transform
def ft(field):
    # inputs:
    # field - The input to compute the Fourier Transform
    ft = np.fft.fft2(field)
    ft = np.fft.fftshift(ft)

    return ft


# Function to compute the Fourier Transform
def ift(field):
    # inputs:
    # field - The input to compute the Fourier Transform
    ift = np.fft.ifft2(field)
    ift = np.fft.fftshift(ift)
    ift = np.fft.fftshift(ift)

    return ift


# Function to get image information
def imgInfo(img):
    # inputs:
    # img - The input img to get the information
    width, height = img.size
    #print(f"Image size: {width} x {height} pixels")

    return width, height


# Function to create a circular mask
def circularMask(width, height, radius, centX, centY, visualize):
    # inputs:
    # width - size image Y
    # height - size image X
    # radius - circumference radius
    # centX - coordinate Y center
    # centY - coordinate X center
    X, Y = np.ogrid[:width, :height]
    mask = np.zeros((width, height))
    circle = np.sqrt((X - centX) ** 2 + (Y - centY) ** 2) <= radius
    mask[circle] = 1

    if visualize:
        imageShow(mask, 'mask')

    return mask


# Function to save an Image
def saveImg(sample, name):
    # inputs:
    # sample - size image Y
    # name - name image
    
    sample = np.abs(sample)
    
    image_data = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255)
    image = Image.fromarray(image_data.astype(np.uint8))
    imageio.imwrite(name, image)
    #image.save('name.png', format="png")

    return


# Function to save a phase image
def savePha(sample, name):
    # inputs:
    # sample - size image Y
    # name - name image
    #sample = intensity(sample, False)
    image_data = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255)
    image = Image.fromarray(image_data.astype(np.uint8))
    imageio.imwrite(name, image)

    return