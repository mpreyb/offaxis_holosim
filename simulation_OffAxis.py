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

# Function to add speckle noise to an object phase
def speckle(object_phase, noise_level, visualization='False'):
    """
    Add Gaussian speckle noise to the object phase.
    
    Parameters:
    - object_phase: The phase of the object (numpy array).
    - noise_level: Standard deviation of the Gaussian noise.
    - visualization: Whether to visualize the noise distribution (str).
    
    Returns:
    - noisy_phase: The noisy object phase (numpy array).
    """
    # Create Gaussian noise
    noise = np.random.normal(0, noise_level, object_phase.shape)
    noisy_phase = object_phase + noise
    
    # Normalize to the range [-π, π]
    noisy_phase = np.mod(noisy_phase + np.pi, 2 * np.pi) - np.pi

    # Visualization of noise distribution if requested
    if visualization == 'True':
        plt.hist(noise.flatten(), bins=30, color='gray')
        plt.title('Speckle Noise Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    return noisy_phase

# Function to convert the noisy object phase to a complex wave
def complexObject(noisy_phase):
    """
    Convert the noisy phase to a complex object wave.
    
    Parameters:
    - noisy_phase: The noisy object phase (numpy array).
    
    Returns:
    - complex_wave: The complex representation of the object wave (numpy array).
    """
    complex_wave = np.exp(1j * noisy_phase)  # E^iφ representation
    return complex_wave

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

# Function to generate an off-axis hologram
def off_axis(object_wave, reference_wave):
    """
    Generate an off-axis hologram by combining object and reference waves.
    
    Parameters:
    - object_wave: The complex object wave (numpy array).
    - reference_wave: The complex reference wave (numpy array).
    
    Returns:
    - hologram: The resulting hologram (numpy array).
    """
    hologram = np.real(object_wave * np.conj(reference_wave))  # Interference pattern
    return hologram

# Function for angular spectrum propagation (general version)
def angularSpectrumog(field, dx, dz, wavelength):
    """
    Propagate a complex field using the angular spectrum method.
    
    Parameters:
    - field: The complex input field (numpy array).
    - dx: Sampling interval in x (float).
    - dz: Propagation distance (float).
    - wavelength: Wavelength of the light (float).
    
    Returns:
    - propagated_field: The propagated field (numpy array).
    """
    # Determine the size of the field and spatial frequencies
    N = field.shape[0]  # Assuming square field
    k = 2 * np.pi / wavelength  # Wave number
    fx = np.fft.fftfreq(N, dx)  # Spatial frequency in x
    fy = np.fft.fftfreq(N, dx)  # Spatial frequency in y
    FX, FY = np.meshgrid(fx, fy)  # Create meshgrid for 2D frequencies

    # Compute the transfer function
    H = np.exp(1j * k * dz * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))

    # Perform Fourier Transform, apply transfer function, and inverse FT
    propagated_field = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(field) * H))
    return propagated_field


# Main function to simulate hologram generation
def hologram(sample, dist, wavelength, noise_level):
    """
    Simulate the generation of a hologram from a sample.
    
    Parameters:
    - sample: The input sample image (numpy array).
    - dist: Distance to the object (float).
    - wavelength: Wavelength of the illumination (float).
    - noise_level: Standard deviation of the Gaussian noise (float).
    
    Returns:
    - hologram: The generated hologram (numpy array).
    """
    # Scale the sample
    scaled_sample = scale_sample(sample, 255)

    # Add speckle noise to the sample
    noisy_phase = speckle(scaled_sample, noise_level, visualization='False')

    # Convert the noisy phase to a complex wave
    complex_wave = complexObject(noisy_phase)

    # Generate the reference wave
    reference_wave = refWave(dist, wavelength)

    # Generate the hologram using off-axis technique
    hologram = off_axis(complex_wave, reference_wave)

    # Propagate the hologram
    propagated_hologram = angularSpectrumog(hologram, dx=1, dz=dist, wavelength=wavelength)

    return propagated_hologram


