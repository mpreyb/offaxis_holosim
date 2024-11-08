'''
    Generation of an out-of-focus DHM hologram database
    
    This script processes a folder of input images, simulates off-axis digital holograms 
    at various propagation distances, and saves the resulting hologram images into an 
    output folder. The simulation uses specified parameters such as wavelength, pixel 
    size, speckle, and image size. The holograms are generated for a range of distances 
    from -10 to 10 mm, allowing for the simulation of out-of-focus holograms, which 
    are essential for deep learning models in Digital Holographic 
    Microscopy (DHM).

    Required .py files:
    - utilities.py: Contains functions such as imageRead and saveImg for handling image operations.
    - simulation_OffAxis.py: Contains the hologram simulation logic (hologram generation).

    Python Version: 3.8.18
    numpy version: 1.24.3

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 17/10/2024
'''


# import libraries
import utilities as ut                  # Custom utility functions used throughout the project.
import simulation_OffAxis as sro        # Custom module for simulating off-axis holograms.
import os                               
import matplotlib.pyplot as plt      
import numpy as np                    

# Simulation parameters (everything should be in the same units)
wavelength = 0.000633      # Illumination wavelength [mm] (for red laser light, 633 nm).
dxy = 0.0065               # Pixel size or the spacing between pixels in the x-y plane (in millimeters).
std_dev = 0.1              # Standard deviation for speckle noise 
disc = 60                  # 
size = 512                 # Output hologram size (512x512 pixels).

# Function to ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory): 
        os.makedirs(directory)       

# Function to process images in the input folder and generate holograms
def process_folder(input_folder, output_folder_holo):
    ensure_dir(output_folder_holo)  # Make sure the output folder exists
    
    # Iterate over all .png files in the specified input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"): # Only process .png files
            image_path = os.path.join(input_folder, filename) # Construct full path to the image file
            
            image = ut.imageRead(image_path, size)   # Read the image using custom utility function, resizing it to (512x512)
            image_label = filename[:-4]              # Extract the file name without the '.png' extension
            
            print("-------------------------------------------------------------------------------")
            print(f"-------------------**PROCESSING IMAGE: {filename}**---------------------------")
            
            # Simulate holograms for distances ranging from -10 to 10 mm, with a step of 1 mm.
            for distance in range(-10, 1, 10):
                print("-------------------------------------------------------------------------")
                print(f"**Simulating hologram with z = {distance} mm...**")  
                
                # Generate the hologram using the custom simulation module.
                hologram = sro.hologram(image, wavelength, dxy, std_dev, distance, disc)
                
                # Create a unique file name for the hologram, indicating the distance
                holoname = f"{image_label}_h_{distance}"
                
                # Define the save path for the hologram image
                save_filename = f"{holoname}.png"                            # Append .png to the hologram name
                save_path = os.path.join(output_folder_holo, save_filename)  # Full path to save the hologram image
                
                # Save the hologram image using a utility function
                ut.saveImg(hologram, save_path)
                
                print(f"**Simulation for z = {distance} mm done!**")  
             
            print("-------------------------------------------------------------------------------")
            print(f"-----------------**DONE PROCESSING IMAGE: {filename}**------------------------")

# Paths for input and output
input_mnist = r'C:\path\to\your\MNIST\database\location'          # Folder where MNIST image files are located.
output_folder_holo = r'C:\path\where\you\want\to\save\holograms'  # Folder to store the simulated holograms.

# Call the process_folder function to begin hologram simulation
process_folder(input_mnist, output_folder_holo)
