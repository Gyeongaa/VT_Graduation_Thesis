# The Repository for VT Graduation Thesis

## About Repository
This repository contains the materials and findings related to my MSc thesis, "Enhanced Disease Classification in Respiratory Sounds: A Transfer Learning Approach Utilizing ICBHI and Coswara Datasets," in Voice Technology at the University of Groningen. The thesis investigated the methods to enhance the accuracy of diagnosing respiratory diseases using transfer learning through data augmentation to overcome the limitation of small datasets. 

In my research, I focus on two main approaches aimed at refining the efficiency and effectiveness of diagnostic models trained on respiratory sound data. The first approach investigates the impact of different layer freezing techniques during transfer learning, exploring how freezing up to the 1st, 2nd, and 3rd layers of a Residual Networks (ResNet)-based model affects the accuracy and loss metrics when training with the ICBHI and Coswara datasets. 
The second approach assesses the comparative performance of two fine-tuning approaches: layer freezing versus classifier adjustment. These hypotheses all demonstrate the successful integration of two breath sound datasets through transfer learning.

The thesis text can be dowloaded from here.
This repository contains:
- The instruction to download used datasets
- The code for preprocessing the ICBHI dataset
- The code for preprocessing the Coswara dataset
- The code for training model using ICBHI dataset
- The fine-tuning code using Coswara dataset

## Dataset
### ICBHI 2017 
Please download the dataset via official site.
You can download official label, split and respiratory samples below link:

```
https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
```


### Coswara
Please download the dataset using below command(It will takes some time):

```
git clone https://github.com/iiscleap/Coswara-Data.git
```

## Training Code



