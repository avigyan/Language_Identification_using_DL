from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
#from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
#from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.regularizers import l2
#from keras import backend as K

NAME = "Lenet"

def create_model(input_shape, config, is_training=True):
	# initialize the model
	weight_decay = 0.001
	
	model = Sequential()

	# if we are using "channels first", update the input shape
	#if K.image_data_format() == "channels_first":
		#inputShape = (numChannels, imgRows, imgCols)

	# define the first set of CONV => ACTIVATION => POOL layers
	model.add(Convolution2D(20, 5, 5, W_regularizer=l2(weight_decay),activation="relu",input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# define the second set of CONV => ACTIVATION => POOL layers
	model.add(Convolution2D(50, 5, 5, W_regularizer=l2(weight_decay), activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# define the first FC => ACTIVATION layers
	model.add(Flatten())
	model.add(Dense(500, W_regularizer=l2(weight_decay), activation="relu"))
	
	# define the second FC layer
	model.add(Dense(config["num_classes"], activation="softmax"))
	
	# if a weights path is supplied (inicating that the model was
	# pre-trained), then load the weights
	#if weightsPath is not None:
		#model.load_weights(weightsPath)
	# return the constructed network architecture
	return model
