import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
import random
import medpy.io
import shutil
import display



"""
All datasets should have the following form for each patient before process
	dataset/
	├─ BraTS20_001/
	│  ├─ BraTS20_001_flair.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_seg.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t1.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t1ce.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t2.(mha, nii, nii.gz, npy)
	├─ BraTS20_002/
	│  ├─ BraTS20_002_flair.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_seg.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t1.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t1ce.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_002_t2.(mha, nii, nii.gz, npy)
	├─ BraTS20_XXX/
	│  ├─ BraTS20_XXX_flair.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_XXX_seg.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_XXX_t1.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_XXX_t1ce.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_XXX_t2.(mha, nii, nii.gz, npy)
	├─ BraTS20_YYY/
	│  ├─ BraTS20_YYY_flair.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_YYY_seg.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_YYY_t1.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_YYY_t1ce.(mha, nii, nii.gz, npy)
	│  ├─ BraTS20_YYY_t2.(mha, nii, nii.gz, npy)

This data processing is made for a U-Net with an input shape of 128x128x128
"""

def load(filepath):
	"""
	Load a brain IRM with either of those extensions (.mha, .nii, .nii.gz, .npy)
	"""
	
	file_ext = filepath.split('.')[-1]
	name = filepath.split('.')[-2].split('/')[-1]
	
	# Load inputs data
	if file_ext == 'mha':
		image_data, image_header = medpy.io.load(filepath)
	elif file_ext == 'nii' or 'nii' + file_ext == 'niigz':
		image_data = nib.load(filepath).get_fdata()
	elif file_ext == 'npy':
		image_data = np.load("filename")
	else:
		print("Make sure to enter a supported file extension (.mha / .nii / .nii.gz / .npy)")
		return -1

	print("File information:")
	print("\t Input file name: {}".format(name))
	print("\t Input file type: {}".format(file_ext))
	print("\t Shape: {}".format(image_data.shape))

	print("----------------------------------------------------------------")
	return image_data



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


def process_images(patient_dir_path, processed_dir_path="", file_ext='nii'):
	"""
	Scaling each image (values from 0 to 1)
	Using MinMaxScaller from sklearn.preprocessing
	"""

	patient = patient_dir_path.split('/')[-1]

	image_flair = load(os.path.join(patient_dir_path, "{}_flair.{}".format(patient, file_ext)))
	image_flair = scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)

	"""
	# Not useful
	test_image_t1 = load(os.path.join(patient_dir_path, "{}_t1.{}".format(patient, file_ext)))
	test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)
	"""

	image_t1ce = load(os.path.join(patient_dir_path, "{}_t1ce.{}".format(patient, file_ext)))
	image_t1ce = scaler.fit_transform(image_t1ce.reshape(-1, image_t1ce.shape[-1])).reshape(image_t1ce.shape)

	image_t2 = load(os.path.join(patient_dir_path, "{}_t2.{}".format(patient, file_ext)))
	image_t2 = scaler.fit_transform(image_t2.reshape(-1, image_t2.shape[-1])).reshape(image_t2.shape)

	if image_flair.shape == image_t1ce.shape == image_t2.shape and image_flair.ndim == image_t1ce.ndim == image_t2.ndim == 3 :
		# Stack 3 most important channel into 1 numpy array to feed the U-Net
		resulting_image = np.stack([image_flair,
									image_t1ce,
									image_t2], axis=3)

		# Crop the image from 240x240x155 to 128x128x128
		resulting_image = resulting_image[56:184, 56:184, 13:141] # Shape 128x128x128x3 (x4 if t1 included)

		np.save(os.path.join(processed_dir_path, "images", patient+".npy"), resulting_image)
		
	else:
		print("Channels don't have the same shape and/or are not conform to the standard IRM dim (X, X, X) for each channel")
		return -1


def process_mask(patient_dir_path, processed_dir_path="", file_ext='nii'):
	""" 
	Mask relabeling 4 -> 3 (label 3 is empty)
	Then croping it to 128x128x128 images, taking out all the blanc usless information (less computing time + less bias for the Neural Network)
	Ratio thing usless for now then Saving it as a numpy array .npy
	"""

	patient = patient_dir_path.split('/')[-1]

	mask = load(os.path.join(patient_dir_path, "{}_seg.{}".format(patient, file_ext)))
	mask = mask.astype(np.uint8)

	mask[mask==4] = 3 
	mask = mask[56:184, 56:184, 13:141] # Shape 128x128x128

	# from a label to 0 to 4 to get the output as probabilities of the label
	# [1, 0, 0, 0] -> 0
	# [0, 1, 0, 0] -> 1
	# [0, 0, 1, 0] -> 2
	# [0, 0, 0, 1] -> 3
	mask = to_categorical(mask, num_classes=4) # num_classes 4 because converted all 4 to 3
	np.save(os.path.join(processed_dir_path, "mask", patient+".npy"), mask)


def process_patient(patient_dir_path, processed_dir_path="", test=False, file_ext='nii'):
	"""
	Processes the images and the mask of a single patient by calling above functions
	"""

	# If True don't process the mask (no ground truth for validation and testing data)
	if test:
		process_images(patient_dir_path, processed_dir_path, file_ext)
	else:
		process_images(patient_dir_path, processed_dir_path, file_ext)
		process_mask(patient_dir_path, processed_dir_path, file_ext)


def processd_dataset(dataset_folder_path, processed_dir_path="processed", test=False, file_ext='nii'):
	"""
	Process an entire dataset
	:param string `dataset_folder_path`: path of the dataset
    :param string `processed_dir_path`: path of the resulting processed dataset
    :param bool val_or_test: True if processing dataset without masks
	:param string file_ext: the file type extension (e.g .nii, .mha ...)
    :return: describe what it returns

	Creates the folders in which the processed images and masks will be stored following this schema:
	
	processed/
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

	# try except possible
	os.mkdir(processed_dir_path)
	os.mkdir(processed_dir_path+"/images")
	os.mkdir(processed_dir_path+"/mask")

	# Processes each patient in the original dataset
	for dirname, patients, filenames in os.walk(dataset_folder_path):
		for patient in patients:
			process_patient(os.path.join(dirname, patient), processed_dir_path, test, file_ext)
			# uncomment the line below if you want to delete the raw data after processing it (convinient for online limitied storage)
			#shutil.rmtree(os.path.join(dirname, patient))



def loadImage(image_dir, image_list):
	"""
	Load images in a given directory for a given image_path list
	:param string image_dir: dir location of the images
	:param list image_list: list of images paths
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
	Omage loader gives batches of images with a size of batch_size
	"""

	L = len(image_list) 

	while True:

		batch_start = 0
		batch_end = batch_size

		while batch_start < L:
			limit = min(batch_end, L)

			X = loadImage(image_dir, image_list[batch_start:limit])
			Y = loadImage(mask_dir, mask_list[batch_start:limit])

			yield(X, Y)

			batch_start += batch_size
			batch_end += batch_size



def process_prediction(prediciton):

	"""
	Get back the original voxel shape and size of 240x240x155 with a prediction generated by the model 
	"""
	
	# Get the highest predicted label
	prediciton = np.argmax(prediciton, axis=4)[0,:,:,:]
	# Replace the segmentation in a 244x244x128 array to match the original voxel of shape 240x240x155
	result = np.zeros([240, 240, 155])
	result[56:184, 56:184, 13:141] = prediciton
	# Revert the label 3 to the original 4
	result[result==3] = 4
	return result


def process_brats_file_submission(filepath, patient, ref):

	# Open the saved prediction
	img = load(filepath).get_fdata()
	img = img.astype(np.float64)
	# Get a nii image with the same affine and header as the ref (ref -> a provided verified image from the challange)
	img = nib.Nifti1Image(img, ref.affine, ref.header)
	# Save the nii.gz with the required */*{ID}.nii.gz
	img.to_filename("BraTS_Submission/{}.nii.gz".format(patient))  # Save as NiBabel file
