# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras import optimizers

from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
K.set_image_dim_ordering('th')
import matplotlib.pyplot as plt
import glob

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
#%%

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 10
# number of epochs to train
nb_epoch = 2


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# input image dimensions
img_rows, img_cols = 200, 200

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
		                border_mode='valid',
		                input_shape=(1, img_rows, img_cols)))
	convout1 = Activation('relu')
	model.add(convout1)
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	convout2 = Activation('relu')
	model.add(convout2)
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
	

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "L":
		image = image.convert("L")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(200, 200))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = preds
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			max = 0
			labels = ['0 - straight and unbowed', '1 - zigzagging up and down', '2 - gently ascending', '3 - gently descending', '4 - climbing abruptly', '5 - falling abruptly', '6 - one sharp peak', '7 - one hil', '8 - one sharp trough', '9 - one valley']
			for x in range(10):
				r = {"label": labels[x], "probability": float(results[0,x])}
				data["predictions"].append(r)
				if results[0,x] >= results[0,max]:	
					max = x

			# indicate that the request was a success
			data["success"] = True
			data["best"] = labels[max]

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
