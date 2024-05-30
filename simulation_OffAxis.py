# -*- coding: utf-8 -*-
"""
Title:-->            simulation off-axis holograms
Description:-->
Authors:-->          Maria Paula Rey, and Raul Castaneda
Emaila:-->           mpreyb@eafit.edu.co
                     racastaneq@eafit.edu.co
Date:-->             04/08/2023
Date last modified:  03/13/2024
University-->        EAFIT University (Applied Optics Group)

Python Version: 3.8.18
numpy version: 1.24.3
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import utilities as ut


# Function to scale the sample
def scale_sample(sample, factor):
    # inputs:
    # sample: image to become in the sample
    minVal = np.min(sample)
    maxVal = np.max(sample)
    object = ((sample - minVal) / (maxVal - minVal))
    object = object * factor

    return object


# Function to add speckle noise
def speckle(object, width, height, std_dev, visualization):
    # inputs:
    # object: object to add the speckle noise
    # wavelength: illumination wavelength
    # width: size image Y
    # height: size image X
    # std_dev: speckle noise standard deviation
    # visualization: True of False
    phaseObject = object * (2 * math.pi) - math.pi

    # Generate random samples from a normal distribution
    speckle = np.random.normal(loc=0, scale=std_dev, size=(width, height))
    speckle = np.clip(speckle, -np.pi, np.pi)
    objSpeckle = phaseObject + speckle

    minVal = np.min(objSpeckle)
    maxVal = np.max(objSpeckle)
    objSpeckle = ((objSpeckle - minVal) / (maxVal - minVal)) * (2 * np.pi) - np.pi

    if visualization == 'True':
        hist, bins = np.histogram(speckle, bins=30, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(bin_centers, hist, color='blue', label='Distribución de la matriz')
        plt.title('Histogram of Speckle distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()
        ut.imageShow(speckle, 'speckle')

    return objSpeckle


def complexObject(object, objSpeckle):
    # inputs:
    # object - object
    # objSpeckle - speckle noise
    objWave = np.exp(1j * objSpeckle)

    return objWave


# Function to generate the reference wave
def refWave(wavelength, dxy, width, height, radius, object):
    # inputs:
    # wavelength - illumination wavelength
    # dxy - pixel pitch
    # width - size image Y
    # height - size image X
    # radius - pupil radius
    
    disc = round(radius)

    X, Y = np.meshgrid(np.arange(-height / 2, height / 2), np.arange(-width / 2, width / 2), indexing='ij')
    fx_0 = height / 2
    fy_0 = width / 2
    k = (2 * math.pi) / wavelength

    min_dist = 2.1 * disc
    max_dist = min_dist + np.random.choice([0, 10, 30, 40, 60])  #for 512x512 images
    #max_dist = min_dist + np.random.choice([0, 10, 20, 30])  #for 256x256 images
    fx = (np.random.choice([min_dist, max_dist])*np.random.choice([-1, 1]))
    fy = (min_dist)

    theta_x = math.asin((fx_0 - fx) * wavelength / (height * dxy))
    theta_y = math.asin((fy_0 - fy) * wavelength / (width * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))

    return ref_wave


# Function to generate off-axis or slightly off-axis holograms
def off_axis(width, height, objWave, reference, disc):
    
    mask = ut.circularMask(width, height, disc, width / 2, height / 2, False)
    
    ft = np.fft.fft2(objWave)
    ft = np.fft.fftshift(ft)
    
    # Apply the mask in the frequency domain
    objWave2 = ft * mask
    
    # Perform the inverse Fourier transform
    ift = np.fft.ifft2(objWave2)
    ift = np.fft.fftshift(ift)
    objWaveFil = np.fft.fftshift(ift)
    
    # Combine the object and reference waves to form the hologram
    holo = np.real((reference + objWaveFil) * np.conj(reference + objWaveFil))
    return holo

def angularSpectrumog(field, width, height, wavelength, distance, dxy):
    # Function to diffract a complex field using the angular spectrum approximation
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx,dy - sampling pitches
    
    k = 2 * np.pi / wavelength 
    
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dxy * M)
    dfy = 1 / (dxy * N)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
    
    phase = np.exp(1j * distance * 2 * np.pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
    
    tmp = field_spec * phase

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
    
    print(f'{np.min(out)}, {np.min(out)}')

    return out


# main function
def hologram(sample, wavelength, dxy, std_dev, distance, disc):
    # inputs:
    # sample -
    # wavelength - illumination wavelength
    # dxy - pixel pitch
    # std_dev - speckle noise standard deviation
    # propagation distance
    
    width, height = ut.imgInfo(sample)
    object = ut.binarize(sample)
    object = scale_sample(object, 1)
    
    # add speckle noise
    sampleSpeckle = speckle(object, width, height, std_dev, visualization='False')

    # complex object wave
    objWave = complexObject(object, sampleSpeckle)

    # propagation
    if distance != 0:
        objWave = angularSpectrumog(objWave, width, height, wavelength, distance, dxy)
    
    #ut.imageShow(ut.intensity(objWave, True), f'Propagated object at {distance}')
    reference = refWave(wavelength, dxy, width, height, disc, objWave)
    
    # Hologram simulation
    holo = off_axis(width, height, objWave, reference, disc)

    return holo


