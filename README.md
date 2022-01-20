# Unsupervised Spiking Neuron Model (SNM) for spike-based reconstruction
Demo code of Unsupervised Spiking Neuron Model (SNM) for spike-based reconstruction

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [SNM-rec](#snm-rec)
	* [Requirements](#requirements)
	* [Folder Structure](#folder-structure)
	* [Network](#network)
	* [Usage](#usage)
	* [Dataset](#dataset)
	* [License](#license)

<!-- /code_chunk_output -->

## Requirements
* Python (2.7 recommended)
* Brian2 (2.2.2.1 recommended)
* Matlab (2018a recommended)


## Folder Structure
  ```
  Unsupervised-SNM-Reconstruction/
  │
  ├── rec——main.py - evaluation of the model
  │
  ├── motioncut.m - matlab interface for motion cut
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models
  │   ├── model.py
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```



## Network
The code in this repo is an SNM example, we provide following implementation：

![Spiking Neural Model](https://github.com/LinZhu111/Fast-SNM-Reconstruction/blob/main/02783.jpg?raw=true)

## Usage

try `python rec_main.py -i path_to_dat_files -m camera_moving -t 200 -o path_to_save_data` to run code.

## Dataset
The testing dataset is available at
- [PKU-Spike-Recon Dataset](https://www.pkuml.org/resources/pku-spike-recon-dataset.html)

## Citation
If you find DenseNet useful in your research, please consider citing:
Ultra-high Temporal Resolution Visual Reconstruction from a Fovea-like Spike Camera via Spiking Neuron Model. TPAMI 2022
Retina-like Visual Image Reconstruction via Spiking Neural Model. CVPR2020

## License
This project is licensed under the MIT License. See  LICENSE for more details
