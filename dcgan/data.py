import re
import csv
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
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
    attr_name = []
    attr_dictionary = {}
    f = open(root + '\list_attr_celeba.csv')
    rdr = csv.reader(f)
    for idx, line in enumerate(rdr):
        if idx == 0:
            for i, attr_name in enumerate(line):
                attr_name.append(attr_name)
        else:
            img_name = line[0]
            for i, attr_value in enumerate(line):
                if i == 0:
                    pass
                if attr_value == 1:
                    if attr_name[i] in attr_dictionary:
                        attr_dictionary[attr_name[i]].append(img_name)
                    else:
                        attr_dictionary[attr_name[i]] = img_name
    return attr_dictionary

def get_image_list_by_attr(attr_name):
    attr_dic = read_attr_label()
    return attr_dic[attr_name]

def get_image_list_by_common_attr(include_attr, exclude_attr, num_image = 0):
    attr_dic = read_attr_label()
    image_set = set()
    for include_img in include_attr:
        image_set.add(attr_dic[include_img])
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