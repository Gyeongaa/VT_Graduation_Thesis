# The Repository for VT MSc Graduation Thesis🎓
## 📂About Repository
This repository contains the materials and findings related to my MSc thesis, **"Enhanced Disease Classification in Respiratory Sounds: A Transfer Learning Approach Utilizing ICBHI and Coswara Datasets"** in Voice Technology at the University of Groningen. The thesis investigated the methods to enhance the accuracy of diagnosing respiratory diseases using transfer learning through data augmentation to overcome the limitation of small datasets. 

In my research, I focus on two main approaches aimed at refining the efficiency and effectiveness of diagnostic models trained on respiratory sound data. The first approach investigates the impact of different layer freezing techniques during transfer learning, exploring how freezing up to the 1st, 2nd, and 3rd layers of a Residual Networks (ResNet)-based model affects the accuracy and loss metrics when training with the ICBHI and Coswara datasets. 
The second approach assesses the comparative performance of two fine-tuning approaches: layer freezing versus classifier adjustment. These hypotheses all demonstrate the successful integration of two breath sound datasets through transfer learning.

The thesis text can be dowloaded from **here**. (The link will be updated after getting reviewed by supervisors in July)


📌**This repository contains:**
- The instruction to download used datasets
- The code for preprocessing the ICBHI dataset: `image_dataloader2.py`, `utils.py`
- The code for preprocessing the Coswara dataset: `image_dataloader3.py`, `utils.py`
- The code for training model using ICBHI dataset: `train2.py`
- The fine-tuning code using Coswara dataset: `train3.py`
- The code for the used ResNet34 model: `network_cnn.py`, `network_hybrid.py` (Please create `nets` folder and then put them inside when you run the codes to avoid errors)
- The used labels, splits file: please refer `lables`, `splits` folders
- The pre-trained model: `model2.pth` and the extracted models during fine-tuning process (As the models' size is over 25mb, the google drive link is introduced. You can download the model through the link in `models` folder😁)
- The instruction how to run the codes

  

## Step1: Downlaod Datasets
### ICBHI 2017 
Please download the dataset via official site.
You can download official label, split and respiratory samples below link:

```
https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
```


### Coswara
Please download the dataset using below command and follow their detail instruction to extract data from compressed files.
_It might take some time._

```
git clone https://github.com/iiscleap/Coswara-Data.git
```

## Step2: Set Up Work Environment🔨
All tasks were performed on the [Hábrók](https://www.rug.nl/society-business/centre-for-information-technology/research/services/hpc/facilities/habrok-hpc-cluster?lang=en)
; High Performance Computing (HPC) Cluster at the University of Groningen. For the training of the models, NVIDIA A100 GPUs within this HPC environment were used. 

First install required libraries by executing the following command:
```
pip install -r requirements.txt
```

If you are working on **Hábrók**, you also need to follow these commands:

```
module load torchvision/0.13.1-foss-2022a
module load matplotlib/3.5.2-foss-2022a
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
```

⚠️Additionally, it is possible that some libraries are missing from requirements.txt during debugging. In this case, please install the most recent version of the library.


### Step3: Run Code💻

#### 1️⃣ Create the pre-trained model:

```
python3 train2.py
          --data_dir /PATH/TO/DATASET \\
          --label_file /PATH/TO/LABEL_FILE \\
          --split_file /PATH/TO/SPLIT_FILE \\
          --model_path /PATH/TO/MODEL \\
          --lr 1e-5 \\
          --batch_size 64 \\
          --num_epochs 20 \\
```

#### 2️⃣ Fine-tuning stage (Strategy 1):
```
python3 train3.py
--data_dir /PATH/TO/DATASET \\
          --label_file /PATH/TO/LABEL_FILE \\
          --split_file /PATH/TO/SPLIT_FILE \\
          --checkpoint /PATH/TO/MODEL/CHECKPOINT \\
          --model_path /PATH/TO/MODEL(Saved) \\
          --freeze_up_to (choose from 1 to 3) \\
          -lr 1e-4 \\
          --batch_size 64 \\
          --num_epochs 20
```

#### 3️⃣ Fine-tuning stage (Strategy 2):

```
python3 train3.py
--data_dir /PATH/TO/DATASET \\
          --label_file /PATH/TO/LABEL_FILE \\
          --split_file /PATH/TO/SPLIT_FILE \\
          --checkpoint /PATH/TO/MODEL/CHECKPOINT \\
          --model_path /PATH/TO/MODEL(Saved) \\
          --freeze_up_to 4 \\
          -lr 1e-4 \\
          --batch_size 64 \\
          --num_epochs 20 \\
```

© 2024. Soogyeong Shin. all rights reserved.
