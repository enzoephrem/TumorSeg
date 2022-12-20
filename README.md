# Segmentation

In this repo you will find any code that was necessary to achieve the segmentation of a brain tumor.
From displaying any type of brain MRI data, to processing it.
Furthermore you will find the code of the model used.
A small notebook will be included to walk through the code and show an exemple of how it could be used using the **[BraTS2020 Dataset (Training + Validation)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)** dataset.


## Exemple

In the [notebook](main.ipynb) you will find an exemple of how to train a neural network to perform segmentation.
Before going into it, make sure to have your [Kaggle API]((https://www.kaggle.com/docs/api)) key at hand.

## [Demo](demo)
Couple of demo file that treats different cases.
## Logs

[Logs](logs) we're stored using a tensorboard callback during the training.

use ``` tensorboard --logdir='log' ```

It will launch the tensorboard instance

## Ressources 

Included in the ressources folder, 3 hist plots of the Dice scores from the brats challenge countinious challenge 2021