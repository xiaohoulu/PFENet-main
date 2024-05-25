# PFENet
## <center>PFENet: Progressive Feature Enhancement Network for Automated Colorectal Polyp Segmentation
Authors: Guanghui Yue, Houlu Xiao

### 1. Preface
This repository provides code for "Progressive Feature Enhancement Network for Automated Colorectal Polyp Segmentation" IEEE TASE-2024. ( [paper]:() ) 

if you have any questions about our paper, feel free to contact me. And if you are using PraNet or evaluation toolbox for your research, please cite this paper
### 2. Overview
#### 2.1. Framework Overview
![PFENet](\image\Network.jpg)
#### 2.2. Qualitative Results
![Qualitative Results](\image\Fig6.jpg)
### 3. Proposed Baseline
#### 3.1. Testing
The testing experiments are conducted using PyTorch with a single GeForce RTX TITAN GPU of 24 GB Memory.
##### 1.Configuring your environment (Prerequisites):

Creating a virtual environment in terminal: conda create -n PraNet python=3.8.

Installing necessary packages: PyTorch 2.0

##### 2.Downloading necessary data:

downloading testing dataset and move it into ./data/TestDataset/, which can be found in this [BaiduNetDisk](https://pan.baidu.com/s/1DWRuou5HV3BAuLHNOJtHZw?pwd=0hiy). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).

downloading training dataset and move it into ./data/TrainDataset/, which can be found in this [BaiduNetDisk](https://pan.baidu.com/s/1riid24WRmu9fT6hJSGXcXw?pwd=3ujq). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

downloading testing weights and move it into ./snapshots/mynetwork.pth, which can be found in this [BaiduNetDisk](https://pan.baidu.com/s/1HB99b3A8NtD9zXAjUcAAmQ?pwd=826p).

##### 3.Testing Configuration:

After you download all the pre-trained model and testing dataset, just run etest.py to generate the final prediction map and calculate metrics including Dice, IoU, Sm, Em, Fm, MAE: replace your trained model directory (--pth_path).

Just enjoy it!

### 4. Citation

### 5. License
