import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
import random
import shutil

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



def process_images(patient_dir_path, processed_path=""):
	"""
	Scaling each image (values from 0 to 1)
	Using MinMaxScaller from sklearn.preprocessing
	"""

	patient = patient_dir_path.split('/')[-1]

	image_flair = nib.load(os.path.join(patient_dir_path, "{}_flair.nii".format(patient))).get_fdata()
	image_flair = scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)

	"""
	# Not useful
	test_image_t1 = nib.load(os.path.join(patient_dir_path, "{}/{}_t1.nii".format(patient, patient))).get_fdata()
	test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)
	"""

	image_t1ce = nib.load(os.path.join(patient_dir_path, "{}_t1ce.nii".format(patient))).get_fdata()
	image_t1ce = scaler.fit_transform(image_t1ce.reshape(-1, image_t1ce.shape[-1])).reshape(image_t1ce.shape)

	image_t2 = nib.load(os.path.join(patient_dir_path, "{}_t2.nii".format(patient))).get_fdata()
	image_t2 = scaler.fit_transform(image_t2.reshape(-1, image_t2.shape[-1])).reshape(image_t2.shape)

	# Stack 3 most important channel into 1 numpy array to feed the U-Net
	resulting_image = np.stack([image_flair,
								image_t1ce,
								image_t2], axis=3)

	# Crop the image to 128x128x128
	resulting_image = resulting_image[56:184, 56:184, 13:141] # Shape 128x128x128x3 (x4 if t1 included)

	np.save(os.path.join(processed_path, "image", patient+".npy"), resulting_image)


def process_mask(patient_dir_path, processed_path=""):
	""" 
	Mask relabeling 4 -> 3 (label 3 is empty)
	Then croping it to 128x128x128 images, taking out all the blanc usless information (less computing time + less bias for the Neural Network)
	Ratio thing usless for now then Saving it as a numpy array .npy
	"""

	patient = patient_dir_path.split('/')[-1]

	mask = nib.load(os.path.join(patient_dir_path, "{}_seg.nii".format(patient))).get_fdata()
	mask = mask.astype(np.uint8)

	# Change the label 4 to 3
	mask[mask==4] = 3 
	mask = mask[56:184, 56:184, 13:141] # Shape 128x128x128

	mask = to_categorical(mask, num_classes=4)
	np.save(os.path.join(processed_path, "mask", patient+".npy"), mask)


def process_patient(patient_dir_path, processed_path="", val=False):
	"""
	Processes the images and the mask of a single patient by calling above functions
	"""
	if val:
		process_mask(patient_dir_path, processed_path)
		process_images(patient_dir_path, processed_path)
		return True
	else:
		if random.randint(0,10) <= 8: # Get a ~80% chance of getting picked for training
			process_mask(patient_dir_path, processed_path)
			process_images(patient_dir_path, processed_path)
			return True
		return False


def process_dataset(dataset_path=None, processed_path="", val=False):
	"""
	Process an entire dataset with a specific schema
	"""

	if dataset_path is None:
		print("Please enter a dataset path")
		return -1

	# Create the folders in which fill the processed images following schema:
	""" training_processed/
	├─ images/
	│  ├─ BraTS20_Training_001.npy
	│  ├─ BraTS20_Training_002.npy
	│  ├─ BraTS20_Training_XXX.npy
	│  ├─ BraTS20_Training_YYY.npy
	├─ mask/
	│  ├─ BraTS20_Training_001.npy
	│  ├─ BraTS20_Training_002.npy
	│  ├─ BraTS20_Training_XXX.npy
	│  ├─ BraTS20_Training_YYY.npy
	"""
	os.mkdir(processed_path)
	os.mkdir(processed_path+"/images")
	os.mkdir(processed_path+"/mask")

	# Processes each patient in the original dataset
	for dirname, patients, filenames in os.walk(dataset_path):
		for patient in patients: 
			if process_patient(os.path.join(dirname, patient), processed_path, val):
				shutil.rmtree(os.path.join(dirname, patient))



def loadImageBis(image_dir, image_list):
	"""
	Load images in a given directory for a given list
	"""

	images = []

	for i, image_name in enumerate(image_list):
		if (image_name.split(".")[-1] == 'npy'):
			image = np.load(image_dir + image_name)
			images.append(image)

	images = np.array(images)

	return images



def imageLoader(image_dir, image_list, mask_dir, mask_list, batch_size):
	"""
	image loader to give batches of images with a size of batch_size
	"""

	L = len(image_list) 

	while True:

		batch_start = 0
		batch_end = batch_size

		while batch_start < L:
			limit = min(batch_end, L)


			X = loadImageBis(image_dir, image_list[batch_start:limit])
			Y = loadImageBis(mask_dir, mask_list[batch_start:limit])

			yield(X, Y)

			batch_start += batch_size
			batch_end += batch_size

