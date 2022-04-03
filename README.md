
<h1 align="center">
  <b>1D Convolution Network Examples </b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7-ff69b4.svg" /></a>    
</p>


The Convolution Network repository contains code samples for using 1D convolution networks as end-to-end anomaly detection models and classifiers. This repository comes in tandem with my blog post [here](https://twan3617.github.io/machine/learning/2022/03/08/The_1D_Convolution.html) (please have a read first!)
# Dataset
The first data set we will be working with is the publicly available Airbus helicopter dataset found [here](https://www.research-collection.ethz.ch/handle/20.500.11850/415151). The data is provided in two files: the training dataset, consisting entirely of "healthy" training data (no anomaly to be detected), and a validation dataset, containing 50% anomalous and 50% normal data. The focus of this dataset is on building models to learn features that characterise normality. However, since we are interested in applying 1D convolution networks (which are supervised networks and hence require labelled data sets), we will combine the training and validation data together and use n-fold cross-validation to test model performance.

Some example data from Airbus: 

[Image](/assets_images/helicopter_data_example.png)

## Data instructions
Download and include the files on the [Airbus](https://www.research-collection.ethz.ch/handle/20.500.11850/415151) website. The notebooks are designed to extract data out of the /dataset folder (replace the file paths with the appropriate ones on your local system if you wish to run the notebooks yourself).

# Methods
The 1D convolution network uses layers of weighted sliding windows to compute numerical features. By combining multiple layers, max pooling and dense layers to combine features into class predictions, 1D convolution networks leverage the power of deep neural learning to perform categorisation even with noisy time series data that would otherwise be difficult to work with. 

The particular deep conv-net architecture we will be using is this: 

[Network_architecture](/assets_images/1Dconvnet_architecture.jpg)

Please note that this repository is currently under construction!