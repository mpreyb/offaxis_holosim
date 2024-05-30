'''
    Simulation of out-of-focus DHM holograms 
    
    Python Version: 3.8.18
    numpy version: 1.24.3

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 26/03/2024
'''

# import libraries
import utilities as ut
import simulation_OffAxis as sro
import os
import numericalReconstruction as numRec
import matplotlib.pyplot as plt
from pyDHM import utilities as utpyDHM
import numpy as np


# Simulation parameters (everything should be same units)
wavelength = 0.000633      # Illumination wavelength [mm]
dxy = 0.0065
std_dev = 0.1            # std_dev speckle noise
disc = 60
size = 512

#if size == 512:
#    disc = 60
#if size == 256:
#    disc == 30

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_folder(input_folder, output_folder_holo):
    ensure_dir(output_folder_holo)  # Make sure the output folder exists
    
    # Iterate over all .png files in the given folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            
            image = ut.imageRead(image_path, size)
            image_label = filename[:-4]
            
            print("-------------------------------------------------------------------------------")
            print(f"-------------------**PROCESSING IMAGE: {filename}**---------------------------")
            
            for distance in range(-50, 60, 10):
                print("-------------------------------------------------------------------------")
                print(f"**Simulating hologram with z = {distance} um...**")
                
                hologram = sro.hologram(image, wavelength, dxy, std_dev, distance, disc)
                holoname = f"{image_label}_h_{distance}"
                save_filename = f"{holoname}.png"                                          # Create a unique file name
                save_path = os.path.join(output_folder_holo, save_filename)                      # Define the full save path
                ut.saveImg(hologram, save_path)
                
                print(f"**Simulation for z = {distance} um done!**")
             
            print("-------------------------------------------------------------------------------")
            print(f"-----------------**DONE PROCESSING IMAGE: {filename}**------------------------")

input_mnist = r'C:\path\to\your\MNIST\database\location '
output_folder_holo = r'C:\path\where\you\want\to\save\holograms'
process_folder(input_mnist, output_folder_holo)