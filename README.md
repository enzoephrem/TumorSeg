# Segmentation

In this repo you will find any code that was necessary to achieve the segmentation of a brain tumor.
From displaying any type of brain MRI data, to processing it.
Furthermore you will find the code of the model used.
A small notebook will be included to walk through the code and show an exemple of how it could be used using the kaggle BraTS 2020 (validation + training) dataset.


## Note

Note that any demo should be launched within the demo folder 

### 00_demo

Shows the process of a patient MRI till segmentation
### 01_demo

Shows the compareson of the mask and the predicted mask

### display_demo

Exemples of how the display functions are used
## Logs

Logs of the trained model can be found in the logs folder

``` tensorboard --logdir='log' ```

It will launch the tensorboard instance

## Ressources 

Included in the ressources folder, 3 hist plots of the Dice scores from the brats challenge countinious challenge 2021