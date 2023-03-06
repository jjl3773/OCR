import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im_folder_path = "../data/ctw-test-01-of-07"
train_path = "../data/train.json"

# doc: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader
class OCR_Dataset(Dataset):
    def __init__(self, image_path, label_file, transform=None) -> None:
        self.image_path = image_path
        self.label_file = label_file
        self.transform = transform
        self.label_json = json.load(open(train_path))

    def __getitem__(self, index):
        # for now, just put os.walk here
        # if including more than one training library, need to iterate
        # until we get to the index we want
        # os.walk walks over folders, not individual files
        image_list = next(os.walk(self.image_path))[2]
        print(len(image_list))
        curr_im_path = image_list[index]
        print(curr_im_path)
        # PIL image format is W H C
        x = Image.open(im_folder_path + "/" + curr_im_path)
        if self.transform:
            x = self.transform(x)
        x = torch.div(x, 255)
        # annotations is a double nested array with groups of 4:
        il = self.label_json["annotations"][index // 4][index % 4]
        bbox = il["adjusted_bbox"]
        y = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3], ord(il["text"])])
        return x, y

    def __len__(self):
        return len(self.label_json["annotations"])


def load_dataset():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.Resize((512, 512)),
        transforms.PILToTensor()
    ])
    dataset = OCR_Dataset(im_folder_path, train_path, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # images, labels = next(iter(dataloader))
    # print(images.shape)
    # print(len(labels))
    # return images, labels
    print("len(dataset): " + str(len(dataset)))
    return dataset
