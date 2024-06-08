#!/usr/bin/python
# Author: Soogyeong Shin
# This code is for fine tuning strategy 1 with ICBHI dataset
import os
import random
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

import librosa
from tqdm import tqdm

from utils import *
from utils import read_split_file

class image_loader2(Dataset):
    def __init__(self, data_dir, label_file, split_file_path, train_flag=True, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.train_flag = train_flag
        self.audio_files = []
        self.labels = {}
        self.load_labels(label_file)
        self.train_files, self.test_files = read_split_file(split_file_path)
        self.load_audio_files()

        # reset audio preprocessing parameters
        self.sample_rate = 4000
        self.n_mels = 64
        self.nfft = 256
        self.hop_length = self.nfft // 2
        self.f_max = 2000

    def load_labels(self, label_file):
        with open(label_file, 'r') as file:
            for line in file:
                try:
                    patient_id, disease = line.strip().split('\t')  # seperate by tap
                    self.labels[patient_id] = disease
                except ValueError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")

    def load_audio_files(self):
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.wav'):
                patient_id = file_name.split('_')[0]
                if patient_id in self.labels:
                    if self.train_flag and file_name.split('.')[0] in self.train_files:
                        self.audio_files.append((os.path.join(self.data_dir, file_name), self.labels[patient_id]))
                    elif not self.train_flag and file_name.split('.')[0] in self.test_files:
                        self.audio_files.append((os.path.join(self.data_dir, file_name), self.labels[patient_id]))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        file_path, label = self.audio_files[index]
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, n_fft=self.nfft, hop_length=self.hop_length, fmax=self.f_max)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Define the target size; which means the longest duration !
        target_size = 652  

        # Pad or truncate the spectrogram to the target size
        pad_size = target_size - mel_spec.shape[1]
        if pad_size > 0:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_size)), mode='constant', constant_values=(0, 0))
        elif pad_size < 0:
            mel_spec = mel_spec[:, :target_size]

        # Ensure mel_spec is correctly converted to a 3-channel image
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec), axis=0)  
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)

        if self.transforms:
            mel_spec = self.transforms(mel_spec)

        label_mapping = {'Healthy': 0, 'URTI': 1, 'COPD': 2, 'Bronchiectasis': 3, 'Pneumonia': 4, 'Bronchiolitis': 5, 'Asthma': 6, 'LRTI': 7} #class classification
        label = label_mapping.get(label, -1)  # Default to -1 if label is not found

        return mel_spec, label

if __name__ == '__main__':
    # below directory paths are used for my case. please change the path
    split_file_path = '/scratch/s5661285/ICHBI/RespireNet/data/official_split2.txt' 
    data_dir = '/scratch/s5661285/ICHBI/RespireNet/data/icbhi_dataset/audio_text_data/'
    label_file = '/scratch/s5661285/ICHBI/RespireNet/data/patient_diagnosis.txt'
    
    train_loader = image_loader2(data_dir=data_dir, label_file=label_file, split_file_path=split_file_path, train_flag=True)
    test_loader = image_loader2(data_dir=data_dir, label_file=label_file, split_file_path=split_file_path, train_flag=False)
    
    for data, label in train_loader:
        print(data.shape, label)
    for data, label in test_loader:
        print(data.shape, label)