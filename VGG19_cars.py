'''
Vehicle detection NN

This script is a straight-forward implementation for the purposes of experimentation ONLY.

Its loads up VGG19 pre-trained on Imagenet

https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5

The outputs are configured to detect various sorts of vehicles. 

To use the script pass in a filename string referring to an image in the same folder as the script.

'''

# Import libraries
import sys
from sys import argv
import numpy as np
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image


# Load variables, parameters and constants
THRESH = 0.01 # This sets the minimum softmaxed prob for a class to be accepted as true
TRUE_CLASSES = [468,436, 479,475,654,656,705,734,751,779,817,829,511,575] # Class to text mapping is from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
TOP_N = 10 # This sets how many of the top predictions to search for true classes

# Try and load image
try:
	_, filename = argv
	img = image.load_img(filename, target_size=(224, 224))
except Exception as e:
	raise IOError("Ooops! Looks like there was a problem loading the image!")

# Preprocess image
img = image.img_to_array(img)
img_input = preprocess_input(np.expand_dims(img, axis=0))

# Import pre-trained network
VGG19 = VGG19(weights='imagenet')
print('loaded')
def return_prediction(network, img):
	'''
	This function detects vehicle presence in an image

	args: 
		img numpy array of preprocessed image
		network Keras network with 1000 outputs relating to Imagenet Classes
	returns:
		bool True or False for a vehicle detected
	stdout:
		True or False as above	

	'''
	# Retrieve predictions for all classes
	predictions = network.predict(img_input)[0]

	# Check if any top predictions meet threshold requirements
	top_N_classes = np.argsort(predictions)[-TOP_N:]
	top_N_probs = np.sort(predictions)[-TOP_N:]

	for i in range(TOP_N):
		if top_N_classes[i] in TRUE_CLASSES and top_N_probs[i] > THRESH:
			sys.stdout.write('True')
			return True
		
	sys.stdout.write('False')
	return False
	
if __name__ == '__main__':
	return_prediction(VGG19, img_input)