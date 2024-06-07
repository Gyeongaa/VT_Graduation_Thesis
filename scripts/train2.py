#!/usr/bin/python                                                       
# Author: Soogyeong Shin      
                                                                    
import os
import itertools
import argparse
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from image_dataloader2 import image_loader2
from sklearn.metrics import confusion_matrix, accuracy_score

# load external modules
from utils import *
from image_dataloader2 import *
from nets.network_cnn import *
from nets.network_cnn import model2
#from nets.network_hybrid import *
from sklearn.metrics import confusion_matrix, accuracy_score
print ("Train import done successfully")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification System')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--label_file', type=str, required=True, help='File containing disease labels')
    parser.add_argument('--split_file', type=str, required=True, help='File containing train/test split information')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_path', type=str, help='Model saving directory')
    args = parser.parse_args()

    # Data transformation
    transforms = Compose([
        Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  # Normalize for already tensor data
    ])

    # Initialize the dataset
    train_dataset = image_loader2(data_dir=args.data_dir, label_file=args.label_file, split_file_path=args.split_file, train_flag=True, transforms=transforms)
    test_dataset = image_loader2(data_dir=args.data_dir, label_file=args.label_file, split_file_path=args.split_file, train_flag=False, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    num_classes = 7
    model = model2()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def compute_accuracy(outputs, labels):
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)
        return corrects.double() / len(labels)

    # Training and validation loop
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)

        val_epoch_loss = val_running_loss / len(test_loader)
        val_epoch_acc = val_running_corrects.double() / len(test_loader.dataset)

        print(f'Epoch {epoch + 1}/{args.num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}')

    # Final evaluation on the test set
    model.eval()
    test_running_loss = 0.0
    test_running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()
            test_running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_acc = test_running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.4f}')

    # Save the trained model
    if args.model_path:
        torch.save(model.state_dict(), os.path.join(args.model_path, 'model_split_changed.pth'))
        print(f'Training completed and model saved as {os.path.join(args.model_path, "model_split_changed.pth")}')
    else:
        print('Model path not specified, model not saved.')

    def evaluate_model(model, data_loader, device):
        model.eval()  # Set the model to evaluation mode
        true_labels = []
        predictions = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.view(-1).tolist())
                true_labels.extend(labels.view(-1).tolist())
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')  # You can change average mode
        conf_matrix = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions)

        print("Accuracy:", accuracy)
        print("F1 Score (Weighted):", f1)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", report)

        return accuracy, f1, conf_matrix, report
    
    final_accuracy, final_f1, conf_matrix, class_report = evaluate_model(model, test_loader, device)

    # Optionally save the metrics to a file or print them
    print("Final Test Accuracy:", final_accuracy)
    print("Final Test F1 Score:", final_f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

if __name__ == '__main__':
    main()