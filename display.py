import numpy as np
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import skimage.io as io
import medpy.io
from math import ceil


def loadImage(filename=None):
	"""
	Load a brain IRM or npy array 
	"""

	if filename == None:
		print("Make sure to enter an image filename")
		return -1

	# Load inputs data
	if filename.split('.')[-1] == 'mha':
		print("Input file: mha")
		image_data, image_header = medpy.io.load(filename)
	elif filename.split('.')[-1] == 'nii' or 'nii' + filename.split('.')[-1] == 'niigz':
		print("Input file: nii")
		image_data = nib.load(filename).get_fdata()
	elif filename.split('.')[-1] == 'npy':
		print("Input file: numpy array")
		image_data = np.load("filename")
	else:
		print("Make sure to enter a supported file extension (.mha / .nii / .nii.gz)")
		return -1

	print("File information:")
	print("Shape: {}".format(image_data.shape))
	return image_data
	

def display3DCuts(image_filename=None, seg_filename=None, image_color_map='bone', seg_color_map='jet', seg_alpha=0.5):

	# Loading image files

	image_data = loadImage(image_filename)
	if len(image_data) < 0:
		# Return value of loadign image < 0 -> error occurred
		return -1

	if seg_filename:
		seg_data = loadImage(seg_filename)
		if image_data.shape != seg_data.shape:
			print("Image data and Segmentation data do not have the same shape {}!={}".format(image_data.shape, seg_data.shape))
			return -1


	# Plot with 3 Columns
	fig, ax = plt.subplots(1, 3)
	plt.subplots_adjust(bottom=0.25)

	# Create layers to place widgets
	axlayer_X = plt.axes([0.25, 0.2, 0.30, 0.03])
	axlayer_Y = plt.axes([0.25, 0.15, 0.30, 0.03])
	axlayer_Z = plt.axes([0.25, 0.1, 0.30, 0.03])

	# Get aprox the middle images for each of X Y Z
	half_X = ceil((image_data.shape[0] - 1)/2)
	half_Y = ceil((image_data.shape[1] - 1)/2)
	half_Z = ceil((image_data.shape[2] - 1)/2)

	# Create Slider for each cut
	slider_X = Slider(axlayer_X, "Sagittal Cut", 0, image_data.shape[0] - 1, valinit=half_X, valstep=1)
	slider_Y = Slider(axlayer_Y, "Coronal Cut", 0, image_data.shape[1] - 1, valinit=half_Y, valstep=1)
	slider_Z = Slider(axlayer_Z, "Horizontal Cut", 0, image_data.shape[2] - 1, valinit=half_Z, valstep=1)

	# Set title for each ax
	ax[0].set_title("{}/{}".format(half_X, image_data.shape[0] - 1))
	ax[1].set_title("{}/{}".format(half_Y, image_data.shape[1] - 1))
	ax[2].set_title("{}/{}".format(half_Z, image_data.shape[2] - 1))

	# Update function for each respective slider and it's plot
	def update_X(val):
		layer = slider_X.val
		ax[0].imshow(image_data[layer,:,:], cmap=image_color_map)
		if seg_filename:
			ax[0].imshow(seg_data[layer,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[layer,:,:]>0))
		ax[0].set_title("{}/{}".format(layer, image_data.shape[0] - 1))
		fig.canvas.draw_idle()

	def update_Y(val):
		layer = slider_Y.val
		ax[1].imshow(image_data[:,layer,:], cmap=image_color_map)
		if seg_filename:
			ax[1].imshow(seg_data[:,layer,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,layer,:]>0))
		ax[1].set_title("{}/{}".format(layer, image_data.shape[1] - 1))
		fig.canvas.draw_idle()

	def update_Z(val):
		layer = slider_Z.val
		ax[2].imshow(image_data[:,:,layer], cmap=image_color_map)
		if seg_filename:
			ax[2].imshow(seg_data[:,:,layer], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,layer]>0))
		ax[2].set_title("{}/{}".format(layer, image_data.shape[2] - 1))
		fig.canvas.draw_idle()


	# Update plot when detected interaction with slider
	slider_X.on_changed(update_X)
	slider_Y.on_changed(update_Y)
	slider_Z.on_changed(update_Z)
	

	# First display
	ax[0].imshow(image_data[half_X,:,:], cmap=image_color_map)
	ax[1].imshow(image_data[:,half_Y,:], cmap=image_color_map)
	ax[2].imshow(image_data[:,:,half_Z], cmap=image_color_map)
	if seg_filename:
		ax[0].imshow(seg_data[half_X,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[half_X,:,:]>0))
		ax[1].imshow(seg_data[:,half_Y,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,half_Y,:]>0))
		ax[2].imshow(seg_data[:,:,half_Z], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,half_Z]>0))


	# Turn off all of the axis
	ax[0].axis('off')
	ax[1].axis('off')
	ax[2].axis('off')

	# Show plot
	plt.show()



