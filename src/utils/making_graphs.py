import argparse
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix(cm_list, store_path, train_dataset, args):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12*4, 12*4))
    class_labels = [key for key in train_dataset.class_dict.keys()]
    for i, (label, matrix) in enumerate(cm_list.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[i//2, i % 2])
        axes[i//2, i % 2].set_xlabel('Predicted')
        axes[i//2, i % 2].set_ylabel('True')
        axes[i//2, i % 2].set_title(label)
        axes[i//2, i % 2].set_xticklabels(class_labels, rotation=45)
        axes[i//2, i % 2].set_yticklabels(class_labels, rotation=45)
        plt.savefig(store_path)
    plt.close()
    print('Saved confusion matrix')