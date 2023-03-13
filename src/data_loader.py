import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im_folder_path = "../data/ctw-trainval-01-of-26"
train_path = "../data/train.jsonl"

chinese_to_index_map = {}
ind = 0

def chinese_to_index(a):
    global ind
    if (a in chinese_to_index_map):
        return chinese_to_index_map[a]
    else:
        chinese_to_index_map[a] = ind
        ind += 1
        return chinese_to_index_map[a]

# doc: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader
class OCR_Dataset(Dataset):
    def __init__(self, image_path, label_file, transform=None) -> None:
        self.image_path = image_path
        self.label_file = label_file
        self.transform = transform
        # self.label_json = json.load(open(train_path))

    def __getitem__(self, index):
        # for now, just put os.walk here
        # if including more than one training library, need to iterate
        # until we get to the index we want
        # os.walk walks over folders, not individual files
        # image_list = next(os.walk(self.image_path))[2]
        # print(len(image_list))
        # curr_im_path = image_list[index]
        # print(curr_im_path)
        # # PIL image format is W H C
        # x = Image.open(im_folder_path + "/" + curr_im_path)
        # if self.transform:
        #     x = self.transform(x)
        # x = torch.div(x, 255)
        # # annotations is a double nested array with groups of 4:
        # il = self.label_json["annotations"][index // 4][index % 4]
        # bbox = il["adjusted_bbox"]
        # y = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3], ord(il["text"])])
        # return x, y

        with open(train_path) as f:
            anno = json.loads(f.readlines()[index])
        annotations = anno["annotations"]
        
        boxes = []
        labels = []
        for anno_group in annotations:
            for annotation in anno_group:
                xmin = annotation["adjusted_bbox"][0]
                ymin = annotation["adjusted_bbox"][1]
                xmax = xmin + annotation["adjusted_bbox"][2]
                ymax = ymin + annotation["adjusted_bbox"][3]

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(chinese_to_index(annotation["text"]))

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        img_batch = []

        img = Image.open(im_folder_path + "/" + anno["image_id"] + ".jpg")
        # img = img.crop(boxes[0])
        # if self.transform:
        #     img = self.transform(img)
        # img = torch.div(img, 255)
        # print(img.shape)
        # print(labels[0])
        for box in boxes:
            copy = img.crop(box)
            if self.transform:
                copy = self.transform(copy)
            copy = torch.div(copy, 255)
            img_batch.append(copy)

        return torch.stack(img_batch), labels

    def __len__(self):
        return 100
        # return len(self.label_json)


def load_dataset():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.Resize((128, 128)),
        transforms.PILToTensor()
    ])
    dataset = OCR_Dataset(im_folder_path, train_path, transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # images, labels = next(iter(dataloader))
    # print(images.shape)
    # print(len(labels))
    # return images, labels
    return dataset
