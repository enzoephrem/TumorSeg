import numpy as np
import nibabel as nib
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import medpy.io



def displaySegDiff(image_filename=None, seg_filename=None, axis='x'):

	"""
	Displays an image of the brain and the segmentated region
	Can move within layers (choosen axis -> to implement)
	"""

	# Make sure param are good
	if image_filename == None or seg_filename == None:
		print("Make sure to enter an image filename and a segmentation filename")
		return -1

	# Load inputs data
	image_data, image_header = medpy.io.load(image_filename)
	seg_data, seg_header = medpy.io.load(seg_filename)

	# Check data shapes
	if image_data.shape[2] != seg_data.shape[2]:
		print("image data and segmentation data don't have the same shape \n {} != {}".format(image_data.shape[2], seg_data.shape[2]))
		return -1

	fig, ax = plt.subplots(2, 1)
	plt.subplots_adjust(bottom=0.25)

	# Create layers to place widgets
	axlayer_previous = plt.axes([0.25, 0.1, 0.15, 0.03])
	axlayer_next = plt.axes([0.50, 0.1, 0.15, 0.03])
	axlayer_layer = plt.axes([0.25, 0.05, 0.15, 0.04])
	axlayer_step = plt.axes([0.50, 0.05, 0.15, 0.03])

	# Create Text boxes to display layers level and steps
	layer_layerbox = TextBox(axlayer_layer, "l=", "0/{}".format(image_data.shape[2] - 1), textalignment="center")
	layer_stepbox = TextBox(axlayer_step, "step=", "1", textalignment="center")

	# Create the previous and next buttons
	previous_button = Button(axlayer_previous, "Previous", image=None, color='0.85', hovercolor='0.95')
	next_button = Button(axlayer_next, "Next", image=None, color='0.85', hovercolor='0.95')


	class Index(object):

		_current = 0
		_step = 1

		def __init__(self, _max):
			self._max = _max

		def prev_image(self, val):
			self._current = max(0, self._current - self._step)
			layer_layerbox.set_val(str(self._current) + "/{}".format(self._max))
			ax[0].imshow(image_data[:,:,self._current], cmap='bone')
			ax[1].imshow(seg_data[:,:,self._current], cmap='jet')
			fig.canvas.draw_idle()

		def next_image(self, val):
			self._current = min(self._current + self._step, self._max)
			layer_layerbox.set_val(str(self._current) + "/{}".format(self._max))
			ax[0].imshow(image_data[:,:,self._current], cmap='bone')
			ax[1].imshow(seg_data[:,:,self._current], cmap='jet')
			fig.canvas.draw_idle()

		def press(self, event):
			if event.key == "right":
				self.next_image(None)
			if event.key == "left":
				self.prev_image(None)
			if event.key == "up":
				self._step += 10
				layer_stepbox.set_val(self._step)
				fig.canvas.draw_idle()
			if event.key == "down":
				self._step -= 10
				if self._step < 1:
					self.step = 1
				layer_stepbox.set_val(self._step)
				fig.canvas.draw_idle()


	callback = Index(image_data.shape[2] - 1)
	fig.canvas.mpl_connect('key_press_event', callback.press)
	previous_button.on_clicked(callback.prev_image)
	next_button.on_clicked(callback.next_image)

	ax[0].imshow(image_data[:,:,0], cmap='bone')
	ax[1].imshow(seg_data[:,:,0], cmap='jet')

	plt.show()