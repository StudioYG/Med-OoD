# Med-OoD
Biomedical segmentation networks easily suffer from the unexpected misclassification between foreground and background objects when learning on limited and imperfect medical datasets. Med-OoD is a data-centric framework that introduces OoD data supervision into the fully-supervised biomedical segmentation to resolve this misclassification bottleneck. Please refer to the paper for more interesting details. 

# Data
Before starting the training or testing sections, the user can choose to either follow Step1 to generate the data from scratch or follow Step2, if the user wants to directly download the prepared data.
## Step1
Create several folders in your workspace
```mkdir Semantic_Labels Image_Patchs Mask_Patchs OoD_Patchs```
## Step2
