import csv
import numpy as np
import cv2

lines = []

with open( 'data/driving_log.csv' ) as csvfile:
    reader = csv.reader(csvfile)
    next( reader )
    for line in reader:
        lines.append(line)


images = []
measurements = []
#print("lines={}".format( lines ) )
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    #print("source_path={} filename={} current_path={}".format( source_path, filename,current_path ) )
    image = cv2.imread( current_path )
    #print("image shape={}".format( image.shape ) )
    images.append( image )
    measurement = float(line[3])
    measurements.append( measurement ) 

X_train = np.array( images )
y_train = np.array( measurements )

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# LeNet
model = Sequential()
# Normalize the data
model.add( Lambda( lambda x: x/255.0 - 0.5, input_shape = ( 160,320,3 ) ) )
model.add( Cropping2D(cropping=((50,20), (0,0) ) ) )
model.add( Convolution2D( 6,5,5, activation='relu' ) )
model.add( MaxPooling2D() )
model.add( Convolution2D( 6,5,5, activation="relu" ) )
model.add( MaxPooling2D() )
model.add( Flatten() )
model.add( Dense(120) )
model.add( Dense(84) )
model.add( Dense(1) )

model.compile( loss='mse', optimizer='adam' )
model.fit( X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
