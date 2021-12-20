# Unsupervised Spiking Neuron Model (SNM) for spike-based reconstruction

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [SNN-rec](#snn-rec)
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
  pytorch-SNNrec/
  │
  ├── rec.py - evaluation of the model
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

![Spiking Neural Model](https://github.com/Crazyspike/snnrec/blob/master/snn_simple.png?raw=true)

## Usage

try `python rec.py -model Spike_Unet_noBN -weight_path /path_to_weight/Spike_Unet_model_simple_noBN-0090.pth -data_path /path_to_data/data.dat -save_path /path_to_save/ -start_time 300 -duration_time 8` to run code.

## Dataset
The testing dataset is available at
- [PKU-Spike-Recon Dataset](https://www.pkuml.org/resources/pku-spike-recon-dataset.html)
## License
This project is licensed under the MIT License. See  LICENSE for more details
