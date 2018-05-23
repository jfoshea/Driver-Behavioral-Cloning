# Behavioral Cloning 

## Overview
Develop a deep learning neural network to replicate (clone) driving behavior. To accomplish this a convolutional neural network (CNN) is developed to correctly steer a car along the road. Specifically the CNN will be trained using data from a specific track. Many aspects of Keras, OpenCV, python, numpy, and matplotlib are used to develop the CNN. The model can be found here  [model.py](https://github.com/jfoshea/Driver-Behavioral-Cloning/blob/master/model.py)

## Installing and Running the Classifier
The following files are included :
- `model.py` Keras implementation of behavioral cloning model 
- `sdc_lib.py` Helper functions for the behavioral cloning model 
- `model.h5` Autonomous driving capture of the behavioral model 
- `run1.mp4` Video capture of autonomous driving of the behavioral model 
- `writeup_report` Detailed writeup of model creation and evaluation 
- `sample_data_set` A sample of images from the data set 
- `model_mse_loss.csv` Keras CSVLogger training output
- `simple_clone_model.py` Very simple keras models for evaluating data set 

The following steps are used to run the model:
1. Install miniconda environment and related packages
    ```
    https://conda.io/miniconda.html
    ```
2. Clone the SDC-BehavioralCloning git repository
    ```  
    $  git clone https://github.com/jfoshea/Driver-Behavioral-Cloning.git
    ```

3. enable cardnd-term1 virtualenv
    ```
    $ source activate carnd-term1
    ```
4. Train the model (Note data set is needed to train a model)
    ```
    $ python model.py
    ```
5. Run the model in autonomous mode in the simulator.
    ```
    $ python drive.py model.h5
    $ Launch simulator select track and click Autonomous Mode
    ```

## Writeup 
A detailed writeup of the behavioral cloning project and challenges are located here [writeup_report] (https://github.com/jfoshea/Driver-Behavioral-Cloning/blob/master/writeup_report.md)

