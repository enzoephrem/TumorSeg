import numpy as np
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import skimage.io as io
from math import ceil
import random


def display2D(brain_data, seg_data=None, brain_color_map='bone', seg_color_map='jet'):
	"""
	Displays random cut random layer for a given 1 sequence voxel
	Option to give segmention (it displays the same cut same layer)
	:param array brain_data: 3D array of brain
	:param array seg_data: 3D array of mask
	:param string brain_color_map: brain color map
	:param string seg_color_map: segmentation color map
	:return int: -1 for error
	"""

	if len(brain_data) < 0:
		return -1
	if brain_data.ndim != 3:
		print("You should provide a 3 dimension array as brain data")
		return -1

	print("Type of brain data {}".format(type(brain_data)))

	if seg_data is not None:
		if len(seg_data) < 0:
			return -1
		if seg_data.ndim != 3:
			print("You should provide a 3 dimension array as seg data")
			return -1
		if brain_data.shape != seg_data.shape:
			print("Image data and Segmentation data do not have the same shape {}!={}".format(brain_data.shape, seg_data.shape))
			return -1

	print("Type of seg data {}".format(type(seg_data.type)))
	print("Brain and Segmentation data have shape {}".format(brain_data.shape))
        
	fig, ax = plt.subplots(1, 2)
        
    
	cut = random.randint(0, 2)

	if cut == 0:
		index = random.randint(0, brain_data.shape[cut])
		if seg_data:
			ax[0].set_title("YZ cut | layer: {}".format(index))
			ax[1].set_title("Segmentation/Mask")
			ax[0].imshow(brain_data[index, :, :], cmap=brain_color_map)
			ax[1].imshow(seg_data[index, :, :], cmap=seg_color_map)
		else:
			plt.title("YZ cut | layer: {}".format(index))
			plt.imshow(brain_data[index, :, :], cmap=brain_color_map)

	elif cut == 1:
		index = random.randint(0, brain_data.shape[cut])
		if seg_data:
			ax[0].set_title("XZ cut | layer: {}".format(index))
			ax[1].set_title("Segmentation/Mask")
			ax[0].imshow(brain_data[:, index, :], cmap=brain_color_map)
			ax[1].imshow(seg_data[:, index, :], cmap=seg_color_map)
		else:
			plt.title("XZ cut | layer: {}".format(index))
			plt.imshow(brain_data[:, index, :], cmap=brain_color_map)

	elif cut == 2:
		index = random.randint(0, brain_data.shape[cut])
		if seg_data:
			ax[0].set_title("XY cut | layer: {}".format(index))
			ax[1].set_title("Segmentation/Mask")
			ax[0].imshow(brain_data[:, :, index], cmap=brain_color_map)
			ax[1].imshow(seg_data[:, :, index], cmap=seg_color_map)
		else:
			plt.title("XY cut | layer: {}".format(index))
			plt.imshow(brain_data[:, :, index], cmap=brain_color_map)
    
	

def display3DCuts(brain_data, seg_data=None, brain_color_map='bone', seg_color_map='jet', seg_alpha=0.5):
	"""
	Display a brain seen from 3 different angles Coronal, Sagittal and Horizontal angles
	with sliders to go through the layers (in 3D)
	:param array brain_data: 3D array of brain 
	:param array seg_data: 3D array of mask
	:param string brain_color_map: brain color map
	:param string seg_color_map: segmentation color map
	:param string seg_alpha: segmentation alpha (defines how much segmentation will be transparent: 1 being oppac)
	:return int: -1 for error
	"""

	if len(brain_data) < 0:
		print("Brain data is empty")
		return -1
	if brain_data.ndim != 3:
		print("You should provide a 3 dimension array as brain data")
		return -1

	print("Type of brain data {}".format(type(brain_data)))

	if seg_data is not None:
		if len(seg_data) < 0:
			print("Segmentation data is empty")
			return -1
		if seg_data.ndim != 3:
			print("You should provide a 3 dimension array as seg data")
		if brain_data.shape != seg_data.shape:
			print("Image data and Segmentation data do not have the same shape {}!={}".format(brain_data.shape, seg_data.shape))
			return -1

	print("Type of seg data {}".format(type(seg_data)))
	print("Brain and Segmentation data have shape {}".format(brain_data.shape))

	# Plot with 3 Columns
	fig, ax = plt.subplots(1, 3)
	plt.subplots_adjust(bottom=0.25)

	# Create layers to place widgets
	axlayer_X = plt.axes([0.25, 0.2, 0.30, 0.03])
	axlayer_Y = plt.axes([0.25, 0.15, 0.30, 0.03])
	axlayer_Z = plt.axes([0.25, 0.1, 0.30, 0.03])

	# Get aprox the middle images for each of X Y Z
	half_X = ceil((brain_data.shape[0] - 1)/2)
	half_Y = ceil((brain_data.shape[1] - 1)/2)
	half_Z = ceil((brain_data.shape[2] - 1)/2)

	# Create Slider for each cut
	slider_X = Slider(axlayer_X, "Sagittal Cut", 0, brain_data.shape[0] - 1, valinit=half_X, valstep=1)
	slider_Y = Slider(axlayer_Y, "Coronal Cut", 0, brain_data.shape[1] - 1, valinit=half_Y, valstep=1)
	slider_Z = Slider(axlayer_Z, "Horizontal Cut", 0, brain_data.shape[2] - 1, valinit=half_Z, valstep=1)

	# Set title for each ax
	ax[0].set_title("{}/{}".format(half_X, brain_data.shape[0] - 1))
	ax[1].set_title("{}/{}".format(half_Y, brain_data.shape[1] - 1))
	ax[2].set_title("{}/{}".format(half_Z, brain_data.shape[2] - 1))

	# Update function for each respective slider and it's plot
	def update_X(val):
		# Get the slider value
		layer = slider_X.val
		# Show the layer with the value on the slider 
		ax[0].imshow(brain_data[layer,:,:], cmap=brain_color_map)
		if seg_data is not None:
			ax[0].imshow(seg_data[layer,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[layer,:,:]>0))
			# Change the title to the current layer
		ax[0].set_title("{}/{}".format(layer, brain_data.shape[0] - 1))
		fig.canvas.draw_idle()

	def update_Y(val):
		layer = slider_Y.val
		ax[1].imshow(brain_data[:,layer,:], cmap=brain_color_map)
		if seg_data is not None:
			ax[1].imshow(seg_data[:,layer,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,layer,:]>0))
		ax[1].set_title("{}/{}".format(layer, brain_data.shape[1] - 1))
		fig.canvas.draw_idle()

	def update_Z(val):
		layer = slider_Z.val
		ax[2].imshow(brain_data[:,:,layer], cmap=brain_color_map)
		if seg_data is not None:
			ax[2].imshow(seg_data[:,:,layer], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,layer]>0))
		ax[2].set_title("{}/{}".format(layer, brain_data.shape[2] - 1))
		fig.canvas.draw_idle()


	# Update plot when interaction detected on slider
	slider_X.on_changed(update_X)
	slider_Y.on_changed(update_Y)
	slider_Z.on_changed(update_Z)
	

	# First display
	ax[0].imshow(brain_data[half_X,:,:], cmap=brain_color_map)
	ax[1].imshow(brain_data[:,half_Y,:], cmap=brain_color_map)
	ax[2].imshow(brain_data[:,:,half_Z], cmap=brain_color_map)
	if seg_data is not None:
		ax[0].imshow(seg_data[half_X,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[half_X,:,:]>0), vmin = 0)
		ax[1].imshow(seg_data[:,half_Y,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,half_Y,:]>0), vmin = 0)
		ax[2].imshow(seg_data[:,:,half_Z], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,half_Z]>0), vmin = 0)


	# Turn off all of the axis
	ax[0].axis('off')
	ax[1].axis('off')
	ax[2].axis('off')

	# Show plot
	plt.show()


def displayPred3DCuts2(brain_data, seg_data, pred_data, image_color_map='bone', seg_color_map='jet', seg_alpha=0.5):

	"""
	Display a brain seen from 3 different angles Coronal, Sagittal and Horizontal angles
	with sliders to go through the layers (in 3D)
	:param array brain_data: 3D array of brain 
	:param array seg_data: 3D array of mask
	:param array pred_data: 3D array of predicted mask
	:param string brain_color_map: brain color map
	:param string seg_color_map: segmentation color map
	:param string seg_alpha: segmentation alpha (defines how much segmentation will be transparent: 1 being oppac)
	:return int: -1 for error
	"""

	if len(brain_data) < 0 or len(seg_data) < 0 or len(pred_data) < 0:
		print("Brain data or Seg data or Pred data is empty")
		return -1
	if brain_data.ndim != 3 or seg_data.ndim != 3 or pred_data.ndim != 3 :
		print("You should provide a 3 dimension array as brain data")
		return -1

	print("Type of Brain data {}".format(type(brain_data)))
	print("Type of Seg data {}".format(type(seg_data)))
	print("Type of Pred data {}".format(type(pred_data)))

	if brain_data.shape != seg_data.shape != pred_data.shape:
		print("Image data and Segmentation data and Pred data do not have the same shape {}!={}!={}".format(brain_data.shape, seg_data.shape, pred_data.shape))
		return -1

	print("Brain and Segmentation data have shape {}".format(brain_data.shape))

	# Plot with 3 Columns
	fig, ax = plt.subplots(2, 3, figsize=(20, 7))
	plt.subplots_adjust(bottom=0.25)

	# Create layers to place widgets
	axlayer_X = plt.axes([0.25, 0.2, 0.30, 0.03])
	axlayer_Y = plt.axes([0.25, 0.15, 0.30, 0.03])
	axlayer_Z = plt.axes([0.25, 0.1, 0.30, 0.03])

	# Get aprox the middle images for each of X Y Z
	half_X = ceil((brain_data.shape[0] - 1)/2)
	half_Y = ceil((brain_data.shape[1] - 1)/2)
	half_Z = ceil((brain_data.shape[2] - 1)/2)

	# Create Slider for each cut
	slider_X = Slider(axlayer_X, "Sagittal Cut", 0, brain_data.shape[0] - 1, valinit=half_X, valstep=1)
	slider_Y = Slider(axlayer_Y, "Coronal Cut", 0, brain_data.shape[1] - 1, valinit=half_Y, valstep=1)
	slider_Z = Slider(axlayer_Z, "Horizontal Cut", 0, brain_data.shape[2] - 1, valinit=half_Z, valstep=1)

	# Set title for each ax
	ax[0][0].set_title("{}/{}".format(half_X, brain_data.shape[0] - 1))
	ax[0][1].set_title("{}/{}".format(half_Y, brain_data.shape[1] - 1))
	ax[0][2].set_title("{}/{}".format(half_Z, brain_data.shape[2] - 1))
	ax[1][0].set_title("{}/{}".format(half_X, brain_data.shape[0] - 1))
	ax[1][1].set_title("{}/{}".format(half_Y, brain_data.shape[1] - 1))
	ax[1][2].set_title("{}/{}".format(half_Z, brain_data.shape[2] - 1))

	# Update function for each respective slider and it's plot
	def update_X(val):
		layer = slider_X.val
		ax[0][0].imshow(brain_data[layer,:,:], cmap=image_color_map)
		ax[0][0].imshow(seg_data[layer,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[layer,:,:]>0))
		ax[1][0].imshow(brain_data[layer,:,:], cmap=image_color_map)
		ax[1][0].imshow(pred_data[layer,:,:], cmap=seg_color_map, alpha=seg_alpha*(pred_data[layer,:,:]>0))
		ax[0][0].set_title("{}/{}".format(layer, brain_data.shape[0] - 1))
		ax[1][0].set_title("{}/{}".format(layer, brain_data.shape[0] - 1))
		fig.canvas.draw_idle()

	def update_Y(val):
		layer = slider_Y.val
		ax[0][1].imshow(brain_data[:,layer,:], cmap=image_color_map)
		ax[0][1].imshow(seg_data[:,layer,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,layer,:]>0))
		ax[1][1].imshow(brain_data[:,layer,:], cmap=image_color_map)
		ax[1][1].imshow(pred_data[:,layer,:], cmap=seg_color_map, alpha=seg_alpha*(pred_data[:,layer,:]>0))
		ax[0][1].set_title("{}/{}".format(layer, brain_data.shape[1] - 1))
		ax[1][1].set_title("{}/{}".format(layer, brain_data.shape[1] - 1))
		fig.canvas.draw_idle()

	def update_Z(val):
		layer = slider_Z.val
		ax[0][2].imshow(brain_data[:,:,layer], cmap=image_color_map)
		ax[0][2].imshow(seg_data[:,:,layer], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,layer]>0))
		ax[1][2].imshow(brain_data[:,:,layer], cmap=image_color_map)
		ax[1][2].imshow(pred_data[:,:,layer], cmap=seg_color_map, alpha=seg_alpha*(pred_data[:,:,layer]>0))
		ax[0][2].set_title("{}/{}".format(layer, brain_data.shape[2] - 1))
		ax[1][2].set_title("{}/{}".format(layer, brain_data.shape[2] - 1))
		fig.canvas.draw_idle()



	# Update plot when detected interaction with slider
	slider_X.on_changed(update_X)
	slider_Y.on_changed(update_Y)
	slider_Z.on_changed(update_Z)

	# First display
	
	ax[0][0].imshow(brain_data[half_X,:,:], cmap=image_color_map)
	ax[0][1].imshow(brain_data[:,half_Y,:], cmap=image_color_map)
	ax[0][2].imshow(brain_data[:,:,half_Z], cmap=image_color_map)

	ax[1][0].imshow(brain_data[half_X,:,:], cmap=image_color_map)
	ax[1][1].imshow(brain_data[:,half_Y,:], cmap=image_color_map)
	ax[1][2].imshow(brain_data[:,:,half_Z], cmap=image_color_map)


	ax[0][0].imshow(seg_data[half_X,:,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[half_X,:,:]>0))
	ax[0][1].imshow(seg_data[:,half_Y,:], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,half_Y,:]>0))
	ax[0][2].imshow(seg_data[:,:,half_Z], cmap=seg_color_map, alpha=seg_alpha*(seg_data[:,:,half_Z]>0))

	ax[1][0].imshow(pred_data[half_X,:,:], cmap=seg_color_map, alpha=seg_alpha*(pred_data[half_X,:,:]>0))
	ax[1][1].imshow(pred_data[:,half_Y,:], cmap=seg_color_map, alpha=seg_alpha*(pred_data[:,half_Y,:]>0))
	ax[1][2].imshow(pred_data[:,:,half_Z], cmap=seg_color_map, alpha=seg_alpha*(pred_data[:,:,half_Z]>0))


	# Turn off all of the axis
	for i in range(2):
		for j in range(2):
			ax[i][j].axis('off')

	# Show plot
	plt.show()