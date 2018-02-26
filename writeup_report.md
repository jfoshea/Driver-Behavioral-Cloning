# Behavioral Cloning 

## Overview
Develop a deep learning neural network to replicate (clone) driving behavior. To accomplish this a convolutional neural network (CNN) is developed to correctly steer a car along the road. Specifically the CNN will be trained using data from a specific track. Several iterative trials are run to evaluate how the model performs its autonomous driving task around the track. The model is improved and the data set is refined until the model successfully navagates the track without going off the road. The model is then run on another track to evaluate if the model has generalized enough to navigate any road succesfully. Many aspects of Keras, OpenCV, python, numpy, and matplotlib are used to develop the CNN. The model can be found here  [model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/model.py)

The goals / steps of this project are the following:
- Data Set Description.
- Data Set Visualization.
- Data Set Pre-processing
- Model Design and Architecture 
- Model Training and Evaluation 
- Conclusion 

## Data Set Description
The supplied data set comes as a file called data.zip file. The structure of the data set is :
- `driving_log.csv` A CSV file where each line has the following format: `center,left,right,steering,throttle,brake,speed`
- `IMG` A subdirectory containing indexed jpg image files that have been gathered from a driving simulator.

## Data Set Visualization
The supplied data set is quite large, the csv file has 8036 lines. The total number of images =  8036 x 3 = 24,108, and 20% (4821 images) are carved out the data set for validation.  A small sample of the data set is shown below which are the center,left,right images from the first two lines of the data set:

![Alt text](sample_data_set/center_2016_12_01_13_30_48_287.jpg#center )
![Alt text](sample_data_set/left_2016_12_01_13_30_48_287.jpg#left)  
![Alt text](sample_data_set/right_2016_12_01_13_30_48_287.jpg#right )
![Alt text](sample_data_set/center_2016_12_01_13_30_48_404.jpg#center )
![Alt text](sample_data_set/left_2016_12_01_13_30_48_404.jpg#left )
![Alt text](sample_data_set/right_2016_12_01_13_30_48_404.jpg#right )

It was also suggested to gather your own data set using the supplied simulator, this would the model to clone your own personal driving behavior. The suggested method of using using the keyboard to drive proved very cumbersome and frustrating. I then tried a game controller and I got better results.  I then appended my own data set to the supplied data set to increase the number of samples. The number of lines increased from 8036 to 9774.  Before I could merge the data sets I had to massage the CSV contents to match the supplied format as follows:
1. Change the path by using this one liner: `sed -i 's#/home/joshea/Desktop/john_driving_data/IMG#IMG#g' driving_log.csv`
2. I appended my driving data to the supplied driving data using `cat john_driving_log.csv >> driving_log.csv`
3. I then aligned the data between the two data sets by removing the spaces between the columns using this vim command `:%s/, /,/g`

## Data Set Pre-processing 
The following techniques were implemented to enhance and pre-process the data set.
- Flipped images are added to augment the data set. This help generalize training and not memorize this specific track.
- Steering measurements are added to augment the measurements. This helps the network to generalize training and not always steer to the left.  
- Images are normalized for zero mean and equal variance using the suggested method `Lambda(lambda x: (x / 255.0) - 0.5)` within keras.
- Images are cropped in Keras to exclude skyline and hood of the car which focuses training on the road. 
- Histogram Equalization applied in an attempt to handle shadows. 
 

### Model Design and Architecture

For this project I implemented three CNNs in Keras. The first one was a simple regression network to prove out a simple Keras design can be implemented and trained on a GPU. I then evaluated Lenet5 design and trained the network using the supplied dataset. I then ran this model through the simulator in autonomous mode. The observation was the car was oversteering all the time. I then added steering adjustments to help reduce the exagerated steering and I could see this helped a lot but it wasnt perfect. This was essentially the flow used in the keras lesson to build a simple model. Refer to [simpl_clone_model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/simple_clone_model.py). The Nvidia model was then introduced in the following videos as workable model for behavioral cloning mode I then decided to focus on this architecture as the basis for this project. After a basic implemetation I noticed an immediate improvement in driving stability, but more work was needed to improve the model. 


## Nvidia Model

![Alt text](Nvidia_Model.png#center ) 

The Nvidia model was the model of choice for this project as this was proven by Nvidia as a working model for behavioral cloning, so I decided to implement this CNN in Keras. For this project I seperated out the code into [model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/model.py) and [sdc_lib.py](https://github.com/jfoshea/BehavioralCloning/blob/master/sdc_lib.py). The keras model is implemented in model.py, and I placed all helper functions in sdc_lib.py

The nvidia design used 66x200x3 sized images, but the simulator generates 160x320x3 images. My first inclination was to resize the simulator images to the suggessted Nvidia format as I felt this was optimal for training using the Nvidia model. To accomodate Nvidia sized images I used `model.add( ... ,input_shape = ( 66,200,3 ))` in keras, and used `cv2.resize( )` to resize each image to 66,200,3. The batch_generator also initialized a numpy array to the same dimenrsions. I also had to implement the `cv2.resize()` function in drive.py in order to run the simulator. However I found that after a few unsuccessful trials that the Nvidia sized images didnt seem to provide a noticeable advantage over the 160x320x3 simulator images, so I abandoned this effort and used default image sizes. The Nvidia paper also suggested using YUV colorspace, so I converted each RGB formatted image to YUV using `cv2.cvtColor( img, cv2.COLOR_RGB2YUV )`. Again I didnt see any advantage of YUV over RGB for track1, in fact I had better success with original RGB format.  The final model for this project has the following additions to the Nvidia model.
- Normalization layer: This was added to normalize input images using Lamda function.
- RELU Layers: These activation functions are added after each Convolution layer and act like ramp functions to address vanishing gradient problem.
- Dropout Layer: This was added to avoid overfitting.

The final model with modifications is as follows:

| **Layer** | **Description**                                               
|-----------|--------------- 
| Input | 160x320x3 RGB image 
| Cropping2D | Crops image to 90x320x3 (50 from top, 20 from bottom) 
| Normalization | (pixel / 255.0) - 0.5)
| Convolution2D | 5x5 kernel,2x2 stride, 24 outputs
| RELU | REctified Linear Unit 
| Convolution2D | 5x5 kernel,2x2 stride, 36 outputs
| RELU | REctified Linear Unit 
| Convolution2D | 5x5 kernel,2x2 stride, 48 outputs
| RELU | REctified Linear Unit 
| Convolution2D | 3x3 kernel,1x1 stride, 64 outputs
| RELU | REctified Linear Unit 
| Convolution2D | 3x3 kernel,1x1 stride, 64 outputs
| RELU | REctified Linear Unit 
| Dropout | dropout probability of 0.5 
| Flatten | Input 4x33x64, Output 8448 
| Fully Connected | Input 100, Output 50 
| Fully Connected | Input 50, Output 10 
| Fully Connected | Input 10, Output 1 

### Model Training and Evaluation

The keras model used the [Adam Optimizer](https://arxiv.org/abs/1412.6980v8) with the default learning rate = 0.001, and Mean Squared Error (MSE) was used as the loss function. I initially trained the model for 3 epochs without dropout layer. I noticed the validation loss was increasing for with successive epochs.

**Model without Dropout**

| **Loss** | **Validation Loss**                                               
|----------|--------------------- 
| loss: 0.4756 | val_loss: 0.0736
| loss: 0.0172 | val_loss: 0.0292
| loss: 0.0100 | val_loss: 0.0368

I then experimented with adding a Dropout after each RELU activation layer and the Flatten layer. I could see the validation loss sometimes continually decreased and sometimes fluctuated. Increasing the number of epochs saw the loss continually decrease and the validation loss decrease but still fluctuate. I found that simply adding a dropout layer after the flatten layer was sufficient to help decreasing validation loss most of the time, but there is always some fluctuation.

**Model with Dropout**

| **Loss** | **Validation Loss**                                               
|----------|--------------------- 
| loss: 0.4260 | val_loss: 0.0542
| loss: 0.0237 | val_loss: 0.0251
| loss: 0.0294 | val_loss: 0.0213

### Conclusion
 This was a very intersting project and I spent a lot of time on this. It was frustrating at times between gathering good data and the numerous iterations through the simulator. I did successfully navigate track 1. I didnt have success yet on track 2. The big shadow throws the car off its track. In an attempt to solve this I added back in the  RGB to YUV conversion and then performed histogram equalization on the Y channel. However this needs more work and in fact didnt help for track1. I will investigate this more to figure out how to navigate track-2.

