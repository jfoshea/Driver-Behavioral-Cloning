# Behavioral Cloning 

## Overview
Develop a deep learning neural network to replicate (clone) driving behavior. To accomplish this a convolutional neural network (CNN) is developed to correctly steer a car along the road. Specifically the CNN will be trained using data from a specific track. Several iterative trials are run to evaluate how the model performs its autonomous driving task around the track. The model is improved and the data set is refined until the model successfully navagates the track without going off the road. The model is then run on another track to evaluate if the model has generalized enough to navigate any road succesfully. Many aspects of Keras, OpenCV, python, numpy, and matplotlib are used to develop the CNN. The model can be found here [model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/model.py)

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
- Images are normalized for zero mean and equal variance `Lambda(lambda x: x / 127.5 - 1.0` within keras.
- Images are cropped to exclude skyline and hood of the car which focuses training on the road. 
- Random Brightness are added to augment the data set. Images with different brightness are achieved by first converting images to HSV, scaling up or down the V channel and converting back to the RGB channel. [Reference](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
- Images are finally converted from RGB to YUV as suggested in the Nvidia model.
 

### Model Design and Architecture

For this project I implemented three CNNs in Keras. The first one was a simple regression network to prove out a simple Keras design can be implemented and trained on a GPU. I then evaluated Lenet5 design and trained the network using the supplied dataset. I then ran this model through the simulator in autonomous mode. The observation was the car was oversteering all the time. I then added steering adjustments to help reduce the exagerated steering and I could see this helped a lot but it wasnt perfect. This was essentially the flow used in the keras lesson to build a simple model. Refer to [simpl_clone_model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/simple_clone_model.py). The Nvidia model was then introduced in the following videos as workable model for behavioral cloning mode I then decided to focus on this architecture as the basis for this project. After a basic implemetation I noticed an immediate improvement in driving stability, but more work was needed to improve the model. 


## Nvidia Model

![Alt text](Nvidia_Model.png#center ) 

The Nvidia model [Nvidia Blog Post](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)  was the model of choice for this project as this was proven by Nvidia as a working model for behavioral cloning, so I decided to implement this CNN in Keras. For this project I seperated out the code into [model.py](https://github.com/jfoshea/BehavioralCloning/blob/master/model.py) and [sdc_lib.py](https://github.com/jfoshea/BehavioralCloning/blob/master/sdc_lib.py). The keras model is implemented in model.py, and I placed all helper functions in sdc_lib.py

The nvidia design used 66x200x3 sized images, but the simulator generates 160x320x3 images. My first inclination was to resize the simulator images to the suggessted Nvidia format as I felt this was optimal for training using the Nvidia model. To accomodate Nvidia sized images I used `model.add( ... ,input_shape = ( 66,200,3 ))` in keras, and used `cv2.resize( )` to resize each image to 66,200,3. The batch_generator also initialized a numpy array to the same dimenrsions.I also decided to use ELU(Exponential linear unit) which take care of the Vanishing gradient problem. I had many iterations in trainng the model, and had difficulty in reducing overfitting. I originally experimented with adding Dropouts after every layer which helped somewhat but the validation loss was still oscillating after about 5 epochs.I then added W_regularizers which are weight regularizers and are used to regularize the weights in the neural network, and this defintely helped with over fitting but I had to run for ~30 epochs to achieve this. The Nvidia paper also suggested using YUV colorspace, OpenCV has a function `cv2.cvtColor( img, cv2.COLOR_RGB2YUV )` to accomplish this. I didnt see any major advantage of YUV over RGB for track1.  The final model for this project has the following additions to the Nvidia model.
- Normalization layer: This was added to normalize input images using Lamda function.
- Weight Regularizer Layers.
- ELU Layers.
- Dropout Layers.

The final model with modifications is as follows:

| **Layer** | **Description**                                               
|-----------|--------------- 
| Input | 66x200x3 RGB image 
| Normalization | (pixel / 127.5) - 1.0)
| Convolution2D | 5x5 kernel,2x2 stride, 24 outputs
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.2 
| Convolution2D | 5x5 kernel,2x2 stride, 36 outputs
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.2 
| Convolution2D | 5x5 kernel,2x2 stride, 48 outputs
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.2 
| Convolution2D | 3x3 kernel,1x1 stride, 64 outputs
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.2 
| Convolution2D | 3x3 kernel,1x1 stride, 64 outputs
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.2 
| Flatten | Input 4x33x64, Output 8448 
| Dropout | dropout probability of 0.5 
| Fully Connected | Input 100, Output 50 
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.5 
| Fully Connected | Input 50, Output 10 
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.3 
| Fully Connected | Input 10, Output 1 
| ELU | Exponential Linear Unit 
| Dropout | dropout probability of 0.3 
| linear | Linear Activation Function 

### Model Training and Evaluation

The keras model used the [Adam Optimizer](https://arxiv.org/abs/1412.6980v8) with the default learning rate = 0.001, and Mean Squared Error (MSE) was used as the loss function. I initially trained the model for 3 epochs without dropout layer. I noticed the validation loss was increasing for with successive epochs. As discussed in the model section I added dropouts after every layer and also added W_regulizers along with ELU activations at each layer. After this I could see the validation loss steadily decrease, but I had to increase the number of epochs to 30. I do notice a slight increase in validation loss toward the end. I can add back in the early stop callback to prevent this, but overall the model trains and navigates the track successfully.  I enabled CSVLogger in Keras capture the validation loss during training. The CSV can be found here [model_mse_loss.csv](https://github.com/jfoshea/SDC-BehavioralCloning/blob/master/model_mse_loss.csv)


### Conclusion
 This was a very intersting project and I spent a lot of time on this. It was frustrating at times between gathering good data and the numerous iterations through the simulator. Two key items for success were removing the bias for driving straight, augment the steering angle for left and right cameras, and finding the correct model to avoid overfitting. In the end it was very satisfying to watch the self driving car navigate track several times. [run1.mp4](https://github.com/jfoshea/SDC-BehavioralCloning/blob/master/run1.mp4)

