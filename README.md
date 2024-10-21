# Simulation of Off-axis DHM holograms
This repository presents a computational framework to simulate off-axis holograms, aimed at generating a comprehensive database of focused and defocused off-axis simulated holograms for training deep learning models. By leveraging the MNIST database to represent various microscopic geometries, this framework efficiently produces defocused holograms via the angular spectrum method. The holograms are created by the amplitude superposition of a complex-valued plane object wave with a plane reference wave. The implemented reconstruction algorithm provides in-focus aberration-free phase and amplitude 
reconstructions by minimizing a normalized variance function. This robust tool automates the simulation of the recording process, facilitating mass production of holograms and enhancing the development and training of deep learning models in DHM applications. A total of 30’972 512x512 holograms were simulated, alongside their corresponding amplitude and phase reconstructions, for a total of 92’216 images.


## Downloads
The complete dataset generated using this algortithm can be downloaded at:
https://www.kaggle.com/datasets/mariareyb/simulated-out-of-focus-holograms

## License & Copyright
Copyright 2024 Universidad EAFIT
Licensed under the MIT License; you may not use this file except in compliance with the License.


## Contact
Authors: Maria Paula Rey, Raul Castañeda
Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)
