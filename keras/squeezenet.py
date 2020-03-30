from __future__ import print_function
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

NAME = "SqueezeNet"

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# Modular function for Fire Node
##### this assumes K.image_dim_ordering() == 'tf'
#def fire_module(x, fire_id, squeeze=16, expand=64):
def fire_module(x, fire_id, squeeze=16, expand=64):
	s_id = 'fire' + str(fire_id) + '/'
	c_axis = 3

	x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
	x = Activation('relu', name=s_id + relu + sq1x1)(x)

	left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
	left = Activation('relu', name=s_id + relu + exp1x1)(left)

	right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
	right = Activation('relu', name=s_id + relu + exp3x3)(right)

	x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
	return x


# Original SqueezeNet from paper.

#def get_squeezenet(nb_classes, dim_ordering='tf'):
def create_model(input_shape, config):
	input_img = Input(shape=input_shape)

	x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
	x = Activation('relu', name='relu_conv1')(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

	x = fire_module(x, fire_id=2, squeeze=16, expand=64)
	x = fire_module(x, fire_id=3, squeeze=16, expand=64)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

	x = fire_module(x, fire_id=4, squeeze=32, expand=128)
	x = fire_module(x, fire_id=5, squeeze=32, expand=128)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

	x = fire_module(x, fire_id=6, squeeze=48, expand=192)
	x = fire_module(x, fire_id=7, squeeze=48, expand=192)
	x = fire_module(x, fire_id=8, squeeze=64, expand=256)
	x = fire_module(x, fire_id=9, squeeze=64, expand=256)
	x = Dropout(0.5, name='drop9')(x)
	nb_classes=3
	x = Convolution2D(nb_classes, 1, 1, border_mode='valid', name='conv10')(x)
	x = Activation('relu', name='relu_conv10')(x)
	x = GlobalAveragePooling2D()(x)
	out = Activation('softmax', name='loss')(x)
	model = Model(input=input_img, output=[out])
	return model




if __name__ == '__main__':
	import time
	from keras.utils.visualize_util import plot

	start = time.time()
	model = get_squeezenet(1000)

	duration = time.time() - start
	print("{} s to make model".format(duration))

	start = time.time()
	model.output
	duration = time.time() - start
	print("{} s to get output".format(duration))

	start = time.time()
	model.compile(loss="categorical_crossentropy", optimizer="adam")
	duration = time.time() - start
	print("{} s to get compile".format(duration))

	plot(model, to_file='images/SqueezeNet.png', show_shapes=True)
