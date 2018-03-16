import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from enum import IntEnum

class Camera( IntEnum ): 
    """ Enums for selecting a Camera """
    CENTER = 0
    LEFT   = 1
    RIGHT  = 2

def get_training_data( path ):
    """ Read csv file and extract data """
    image_names = []
    steering_angles = []
    with open( path ) as csvfile:
        reader = csv.reader(csvfile)
        next( reader )
        for center,left,right,steering,throttle,brake,speed in reader:
            image_names.append( [ center.strip(), left.strip(), right.strip() ] )
            steering_angles.append( float( steering ) )
    return image_names, steering_angles

def steering_adjust( camera_select, steering_angle ):
    if camera_select == Camera.LEFT:
        steering_angle = steering_angle + 0.25 
    elif camera_select == Camera.RIGHT:
        steering_angle = steering_angle - 0.25 
    return steering_angle

def random_brightness( img ):
    """ Add random brightness to help with varying lighting conditions  """
    img = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    brightness = .25 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * brightness
    img = cv2.cvtColor( img, cv2.COLOR_HSV2RGB )
    return img

def resize_image( img ):
    """ Crop and Resize to Nvidia format """
    img = img[ 40:-20,: ]
    img = cv2.resize( img, ( 200, 66 ), interpolation=cv2.INTER_AREA )
    return img

def random_flip( img, steering_angle ):
    """ Randomly flip an image and reverse steering """
    if random.randrange( 2 ) == 1:
        img = cv2.flip( img, 1 )
        steering_angle = -steering_angle
    return img, steering_angle 

def rgb_to_yuv( img ):
    """ Convert RGB to YUV format """
    img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    return img

def process_image( img ):
    """ Combine all image processing steps here """
    img = random_brightness( img )
    img = resize_image( img )
    img = rgb_to_yuv( img )
    return img

def batch_generator( X, y, path, batch_size ):
    image_batch = np.zeros( ( batch_size, 66, 200, 3 ), dtype=np.float32 )
    steering_batch = np.zeros( ( batch_size, ), dtype=np.float32 )

    while 1:
        steering_zero_count = 0
        for i in range( batch_size ):
            # Select a batch index and camera view at random 
            batch_index = random.randrange( len( X ) )
            camera_select = random.choice( range( len( Camera ) ) )
            steering_angle = y[ batch_index ]
            steering_angle = steering_adjust( camera_select, steering_angle )

            # Limit zero steering angle to half the batch  
            if abs( steering_angle ) < 0.1:
                steering_zero_count += 1
            if steering_zero_count > ( batch_size * 0.5 ):
                while abs( steering_angle ) < 0.1:
                    batch_index = random.randrange( len( X ) )
                    camera_select = random.choice( range( len( Camera ) ) )
                    steering_angle = y[ batch_index ]
                    steering_angle = steering_adjust( camera_select, steering_angle )
            img = mpimg.imread( path + X[ batch_index ][ camera_select ].split('/')[-1] )
            img = process_image( img )
            img = np.array( img, dtype=np.float32 )
            img, steering_angle = random_flip( img, steering_angle )

            image_batch[ i ] = img
            steering_batch[ i ] = steering_angle

        yield image_batch, steering_batch

