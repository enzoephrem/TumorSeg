import sys
sys.path.append('src/')
import display
import numpy as np
import nibabel as nib
from keras.models import load_model


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Patient with a big tumor
patient = "00153"
# Patient with a small tumor
#patient = "01754"

not_processed_path = "RSNA_ASNR_MICCAI_BraTS2021_ValidationData/BraTS2021_{}/BraTS2021_{}_".format(patient, patient)



# Display data
display.display3DCuts(display.loadImage(not_processed_path+"t1ce.nii.gz"))


# Pre-Processing

image_flair = nib.load(not_processed_path+"flair.nii.gz").get_fdata()
image_flair = scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)

"""
# Not useful
image_t1 = nib.load(filename+"t1.nii.gz").get_fdata()
image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)
"""

image_t1ce = nib.load(not_processed_path+"t1ce.nii.gz").get_fdata()
image_t1ce = scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)

image_t2 = nib.load(not_processed_path+"t2.nii.gz").get_fdata()
image_t2 = scaler.fit_transform(image_t2.reshape(-1, image_t2.shape[-1])).reshape(image_t2.shape)

# Stack 3 most important channel into 1 numpy array to feed the U-Net
processed_image = np.stack([image_flair,
							image_t1ce,
							image_t2], axis=3)


processed_image = processed_image[56:184, 56:184, 13:141] # Shape 128x128x128x3



display.display3DCuts(processed_image[:, :, :, 1])


model = load_model('saved_models/brats_3d_simple_unet_30.h5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})


input_image = np.expand_dims(processed_image, axis=0)

prediction = model.predict(input_image)

# Prediction process

# Get the highest predicted label
prediciton = np.argmax(prediction, axis=4)[0,:,:,:]
# Replace the segmentation in a 240x240x155 array to match the original image shape 240x240x155
result = np.zeros([240, 240, 155])
result[56:184, 56:184, 13:141] = prediciton
# Revert the label 3 to the original 4
result[result==3] = 4

display.display3DCuts(display.loadImage(not_processed_path+"flair.nii.gz"), result)