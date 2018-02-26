import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from enum import Enum

class Camera( Enum ): 
    """ Enums for selecting a Camera """
    CENTER = 0
    LEFT   = 1
    RIGHT  = 2

STEERING_ADJUST = 0.2

def get_training_data( path ):
    """ Read csv file and extract data """
    image_names = []
    steerings = []
    with open( path ) as csvfile:
        reader = csv.reader(csvfile)
        next( reader )
        for center,left,right,steering,throttle,brake,speed in reader:
            #print("center = ",center)
            image_names.append( [ center.strip(), left.strip(), right.strip() ] )
            steerings.append( float(steering) )

    return image_names, steerings

def flip_image( img ):
   """ flip image """
   return cv2.flip( img, 1 )

def rotate_image( img, angle ):
   """ Rotates an image using angle parameter """
   img = img.copy()
   img_mid = tuple( np.array( img.shape[0:2] ) / 2 )
   rot_matrix = cv2.getRotationMatrix2D( img_mid, angle, 1.0 )
   return cv2.warpAffine( img, rot_matrix, img.shape[0:2], flags=cv2.INTER_LINEAR )

def translate_image( img, shift ):
   """ Translate/Shift an image x,y direction """
   img = img.copy()
   trans_matrix = np.float32([ [1,0,shift], [0,1,shift] ])
   return cv2.warpAffine( img, trans_matrix, img.shape[:2] )

def rgb_to_yuv( img ):
   """ Convert RGB to YUV format """
   return cv2.cvtColor( img, cv2.COLOR_RGB2YUV )

def equalize_y_channel( img ):
    """Applies Histogram Equalization Y Channel to enhance contrast and shadows"""
    img = img.copy()
    img = cv2.cvtColor( img, cv2.COLOR_RGB2BGR )
    img_yuv = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    # Equalize the histogram of the Y channel
    img[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return img

def process_sample( img, camera_select, steering, training_mode ):
    """ 
        1. Randomly adjust left and right cameras measurements
        2. Randomly flip the camera image and reverse steering angle
        3. Apply histogram equalization on Y channel 
    """
    if training_mode:
        if camera_select == Camera.LEFT:
            steering = steering + STEERING_ADJUST 
        elif camera_select == Camera.RIGHT:
            steering = steering - STEERING_ADJUST 
        if random.choice( [0,1] ) == 1:
            img = flip_image( img )
            steering = -1.0 * steering 

    #img = equalize_y_channel( img )
    return img, steering

def batch_generator( X_train, y_train, path, batch_size, training_mode ):
    """ Generator function to process data set in batches"""

    num_samples = len( X_train )
    images = np.zeros( ( batch_size, 160,320,3 ) )
    steerings = np.zeros( ( batch_size, ), dtype=np.float32 )

    while 1:
        for i in range( num_samples ):
            image_samples = X_train[i:i+batch_size]
            steering_samples = y_train[i:i+batch_size]
            batch_samples = list( zip( image_samples, steering_samples ) )

            for sample in ( batch_samples ):
                camera_select = 0
                if training_mode and random.choice( [0,1] ) == 1:
                    camera_select = random.choice( range( len( Camera ) ) )

                img = mpimg.imread( path + sample[0][ camera_select ].split('/')[-1] )
                steering = float( sample[1] )

                img,steering = process_sample( img, camera_select, steering, training_mode )

                img = np.array( img )
                images[ i % batch_size ] = img
                steerings[ i % batch_size ] = steering 

            yield images, steerings 
