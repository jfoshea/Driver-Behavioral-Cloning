from sdc_lib import batch_generator, get_training_data, Camera 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import CSVLogger, EarlyStopping
from keras.optimizers import Adam

def get_model( ):
    """ Create Nvidia model for behavioral cloning """
    model = Sequential()
    model.add( Cropping2D( cropping = ( (50,20),(0,0) ) , input_shape = ( 160,320,3 ) ) )
    model.add( Lambda( lambda x: ( x/255.0 ) -0.5 ) )
    model.add( Convolution2D( 24, 5, 5, subsample = ( 2,2 ), activation = 'relu' ) )
    model.add( Convolution2D( 36, 5, 5, subsample = ( 2,2 ), activation = 'relu' ) )
    model.add( Convolution2D( 48, 5, 5, subsample = ( 2,2 ), activation = 'relu' ) )
    model.add( Convolution2D( 64, 3, 3, subsample = ( 1,1 ), activation = 'relu' ) )
    model.add( Convolution2D( 64, 3, 3, subsample = ( 1,1 ), activation = 'relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Flatten() )
    model.add( Dense( 100 ) )
    model.add( Dense( 50 ) )
    model.add( Dense( 10 ) )
    model.add( Dense( 1 ) )

    return model 

def main( ):
    PATH_TO_CSV = '../data/driving_log.csv'
    PATH_TO_IMG = '../data/IMG/'
    batch_size = 32

    # Get the training data from log file, shuffle, and split into train/validation datasets
    X_train, y_train = get_training_data( PATH_TO_CSV )
    X_train, y_train = shuffle( X_train, y_train )
    X_train, X_validation, y_train, y_validation = train_test_split( X_train, y_train, test_size=0.2 )
    print("len(X_train) {} len(X_validation) = {}".format( len(X_train), len(X_validation) ) )


    training_generator = batch_generator( X_train, y_train, PATH_TO_IMG, batch_size, training_mode=True )
    validation_generator = batch_generator( X_validation, y_validation, PATH_TO_IMG, batch_size, training_mode=False )

    model = get_model( )

    #model.compile( optimizer=Adam( lr=0.0001 ), loss="mse" )
    model.compile( optimizer=Adam( lr=0.001 ), loss="mse" )

    # Define callbacks for csv logger and early stop
    csv_logger = CSVLogger( 'model_mse_loss.csv', append=True, separator=';' )
    early_stop = EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto' )

    model.fit_generator(training_generator,
                        samples_per_epoch = len(X_train), 
                        validation_data = validation_generator, 
                        nb_val_samples = len(X_validation), 
                        callbacks=[csv_logger],
                        nb_epoch = 10 ) 

    model.save('model.h5')

if __name__ == "__main__":
    """ Main entry point """
    main( )

