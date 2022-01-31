import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class Cifar10Dataset(Dataset):
    def __init__(self, data, filenames, labels, all_labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_paths = filenames
        self.img_list = data
        self.transform = transform
        self.target_transform = target_transform
        self.all_label = all_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_list[idx]
        label = self.img_labels[idx]
        label_onehot = np.zeros(len(self.all_label))
        label_onehot[self.img_labels[idx]] = 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, label_onehot

def get_cifar10_dataloader(batch_size, shuffle=True):
    trainDataset, testDataset = get_cifar10_dataset('../data/cifar-10-batches-py')
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size,shuffle =shuffle)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size,shuffle=shuffle)
    return trainDataLoader, testDataLoader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar10_dataset(data_dir):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return Cifar10Dataset(cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_label_names),\
           Cifar10Dataset(cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names)