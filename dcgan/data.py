import re
import csv

import numpy
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
from PIL import Image
from config import *

root = 'C:\workspace\data\image\celebA'

def get_celeba(params):
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    dataset = dset.ImageFolder(root=root, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader

def read_attr_label():
    attr_name_list = []
    attr_dictionary = {}
    f = open(root + '\list_attr_celeba.csv')
    rdr = csv.reader(f)
    for idx, line in enumerate(rdr):
        if idx == 0:
            for i, attr_name in enumerate(line):
                attr_name_list.append(attr_name)
        else:
            img_name = line[0]
            if "all_image" in attr_dictionary:
                attr_dictionary["all_image"].append(img_name)
            else:
                attr_dictionary["all_image"] = [img_name]
            for i, attr_value in enumerate(line):
                if i == 0:
                    continue
                if attr_value == '1':
                    if attr_name_list[i] in attr_dictionary:
                        attr_dictionary[attr_name_list[i]].append(img_name)
                    else:
                        attr_dictionary[attr_name_list[i]] = [img_name]
    return attr_dictionary

def get_image_list_by_attr(attr_name):
    attr_dic = read_attr_label()
    return attr_dic[attr_name]

def get_image_list_by_common_attr(include_attr, exclude_attr, attr_dic = None, num_image = 0):
    if attr_dic == None:
        attr_dic = read_attr_label()
    image_set = set()
    for include_img in include_attr:
        image_set = image_set.union(set(attr_dic[include_img]))
    if len(include_attr) == 0:
        image_set = set(attr_dic["all_image"])
    for exclude_img in exclude_attr:
        image_set = image_set - set(attr_dic[exclude_img])
    if num_image != 0:
        return list(image_set)[:num_image]
    return list(image_set)

def get_last_model():
    if os.path.exists("checkpoint") == False:
        return None, -1
    file_list = os.listdir("checkpoint")
    file_list.sort()
    file_list_pth = [file for file in file_list if file.endswith(".pth")]
    if len(file_list_pth) == 0:
        return None, -1
    last_model = torch.load("checkpoint/"+file_list_pth[-1])
    last_epoch = re.findall("\d+", file_list_pth[-1])
    return last_model, int(last_epoch[0])

def get_abspath_dictionary(dataloader):
    abspath_dictionary = {}
    for file_name, _ in dataloader.sampler.data_source.imgs:
        abspath_dictionary[os.path.basename(file_name)] = file_name
    return abspath_dictionary

def get_abspath(filename, abspath_dictionary):
    return abspath_dictionary[filename]

def get_abspath_list(filename_list, abspath_dictionary):
    abspath_list = []
    for filename in filename_list:
        abspath_list.append(abspath_dictionary[filename])
    return abspath_list

def imread_to_tensor(image_path):
    image = Image.open(image_path)
    image_np = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    image_np = np.transpose(image_np, (2, 0, 1))
    image_tensor = torch.tensor(image_np, device=device)
    return image_np #image_tensor

def imread_multiple_to_tensor(image_path_list):
    image_tensor_list = []
    for image_path in image_path_list:
        image_tensor = imread_to_tensor(image_path)
        image_tensor_list.append(image_tensor)
    return torch.tensor(np.array(image_tensor_list), device=device)