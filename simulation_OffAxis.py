'''
    Simulation of out-of-focus DHM holograms 
    
    This code simulates the generation of holograms from a given sample image.
    It includes functions for scaling sample values, adding speckle noise,
    converting the noisy phase to a complex wave, generating a reference wave,
    and creating an off-axis hologram. In this simulation, both the reference wave and object wave
    are plane waves.
    
    The code also implements the angular spectrum method for propagating the hologram to a 
    specified distance (out-of-focus).
    
    Required .py files:
    - utilities.py: Contains functions such as imageRead and saveImg for handling image operations.

    Python Version: 3.8.18
    numpy version: 1.24.3

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 17/10/2024
'''

# Import necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import utilities as ut  # Custom utilities for additional functions

# Function to scale sample values to a specified factor
def scale_sample(sample, factor):
    """
    Scale the pixel values of a sample to the specified factor.
    
    Parameters:
    - sample: The input sample image (numpy array).
    - factor: The scaling factor.
    
    Returns:
    - scaled_sample: Scaled sample image.
    """
    scaled_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * factor
    return scaled_sample



# Function to add speckle noise
def speckle(object, width, height, std_dev, visualization):
    """
    Add speckle noise to the object wave.
    
    Parameters:
    - object: object to add the speckle noise.
    - wavelength: illumination wavelength.
    - width: size image Y.
    - height: size image X.
    - std_dev: speckle noise standard deviation
    - visualization: True of False
    - noise_level: Standard deviation of the Gaussian noise.
    - visualization: Whether to visualize the noise distribution (str).
    
    Returns:
    - objSpeckle: The noisy object phase.
    """

    phaseObject = object * (2 * math.pi) - math.pi

    # Generate random samples from a normal distribution
    speckle = np.random.normal(loc=0, scale=std_dev, size=(width, height))
    speckle = np.clip(speckle, -np.pi, np.pi)
    objSpeckle = phaseObject + speckle

    minVal = np.min(objSpeckle)
    maxVal = np.max(objSpeckle)
    objSpeckle = ((objSpeckle - minVal) / (maxVal - minVal)) * (2 * np.pi) - np.pi
    #print(np.min(objSpeckle), np.max(objSpeckle))

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

# Function to convert the noisy object phase to a complex wave
def complexObject(noisy_phase):

    complex_wave = np.exp(1j * noisy_phase)  # E^iφ representation
    return complex_wave

def complexObject(object, objSpeckle):
    """
    Convert the noisy phase to a complex object wave.
    
    Parameters:
    - objSpeckle: The noisy object phase.
    
    Returns:
    - objWave: The complex representation of the object wave.
    """
    objWave = np.exp(1j * objSpeckle)     # E^iφ representation

    return objWave

# Function to generate a reference wave with random angles
def refWave(dist, wavelength):
    """
    Generate a reference wave for holography.
    
    Parameters:
    - dist: Distance to the object (float).
    - wavelength: Wavelength of the illumination (float).
    
    Returns:
    - reference_wave: The complex representation of the reference wave (numpy array).
    """
    # Generate random angle and compute phase
    theta = np.random.uniform(0, 2 * np.pi)
    reference_wave = np.exp(1j * (2 * np.pi / wavelength * dist * np.cos(theta)))
    return reference_wave

# Function to generate the reference wave
def refWave(wavelength, dxy, width, height, radius, object):

     """
    Generate a reference wave for holography.
    
    Parameters:
    - wavelength: illumination wavelength.
    - dxy: pixel pitch.
    - width: size image Y.
    - height: size image X.
    - radius: pupil radius. 
    
    Returns:
    - ref_wave: The complex representation of the reference wave.
    
    """

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


# Function to generate off-axis hologram
def off_axis(width, height, objWave, reference, disc):
    """
    Generate an off-axis hologram by combining object and reference waves.
    
    Parameters:
    - objWave: The complex object wave.
    - reference: The complex reference wave.
    
    Returns:
    - holo: The resulting hologram.
    """
    
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
    #holo = np.abs(reference + objWaveFil)**2
    return holo


def angularSpectrumog(field, width, height, wavelength, distance, dxy):

     """
    Propagate a complex field using the angular spectrum method.
    
    Parameters:
    - field: The complex input field (numpy array).
    - dx, dy: sampling pitches (float).
    - z: Propagation distance (float).
    - wavelength: illumination wavelength. (float).
    
    Returns:
    - propagated_field: The propagated field (numpy array).
    """

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
    
    return out


# Main function to simulate hologram generation
def hologram(sample, wavelength, dxy, std_dev, distance, disc):
    """
    Simulate the generation of a hologram from a sample.
    
    Parameters:
    - sample: The input sample image (numpy array).
    - distance: Propagation distance (float).
    - wavelength: illumination wavelength (float).
    - std_dev: speckle noise standard deviation (float).
    
    Returns:
    - hologram: The generated hologram (numpy array).
    """
    # Get image size information
    width, height = ut.imgInfo(sample)

    #Binarize MNIST image for sharpness
    object = ut.binarize(sample)

    object = scale_sample(object, 1)
    
    # ut.imageShow(object, 'scale object')
    
    #ut.imageShow(ut.intensity(object, False), 'Sample (Intensity)')

    # add speckle noise
    sampleSpeckle = speckle(object, width, height, std_dev, visualization='False')
    #ut.imageShow(sampleSpeckle, 'sample + speckle')

    # complex object wave
    objWave = complexObject(object, sampleSpeckle)

    # propagation
    if distance != 0:
        objWave = angularSpectrumog(objWave, width, height, wavelength, distance, dxy)
    
    #ut.imageShow(ut.intensity(objWave, False), 'Propagated')
    
    #ut.imageShow(ut.intensity(objWave, True), f'Propagated object at {distance}')
    reference = refWave(wavelength, dxy, width, height, disc, objWave)
    
    # Hologram simulation
    holo = off_axis(width, height, objWave, reference, disc)

    return holo



