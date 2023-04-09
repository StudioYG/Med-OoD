# Med-OoD
Biomedical segmentation networks easily suffer from the unexpected misclassification between foreground and background objects when learning on limited and imperfect medical datasets. Med-OoD is a data-centric framework that introduces OoD data supervision into the fully-supervised biomedical segmentation to resolve this misclassification bottleneck. Please refer to the paper for more interesting details. 

# Data
Before starting the training or testing sections, the user can choose to either follow Step1 to generate the data from scratch or follow Step2, if the user wants to directly download the prepared data.
## Step1
1. Create several folders in your workspace:
```mkdir Semantic_Labels Image_Patchs Mask_Patchs OoD_Patchs```
2. Download the official Lizard dataset from https://warwick.ac.uk/fac/cross_fac/tia/data/lizard and place each folder well in your workspace, then run the script: ```python generate_semantic_labels.py``` to generate the semantic masks in the folder called Semantic_Labels
3. Run another two scripts:```python generate_patches.py``` and ```python generate_unfiltered_oods.py``` to generate the patches in the folders called Image_Patchs, Mask_Patchs and OoD_Patchs 
## Step2
1. Download the prepared data directly from ------ and place the folders including Image_Patchs, Mask_Patchs and OoD_Patchs properly in your workspace. 
# Training
1. ```config.py``` is the configuration file which can be used to modify encoder, architecture and other settings. 
2. ```train.py``` is the training script for baselines and ```train_ood.py``` is the training script when applying Med-OoD to baselines. 
3. To train both versions, i.e., baseline and baseline+Med-OoD, just run the script: ```python main.py```
# Testing
```test.py``` is the testing script which user can use to test the trained models. Unet-VGG11BN and Unet-VGG11BN+Med-OoD trained on Lizard dataset have also been released at 
