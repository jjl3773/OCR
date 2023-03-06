import torch
from typing import List
from data_loader import load_dataset

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch import flatten

from torch.optim import Adam
from torch.nn.functional import cross_entropy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

class OCRNetwork(Module):
    def __init__(self, input_size, output_size):
        super(OCRNetwork, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=input_size, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=781250, out_features=500)
        self.relu3 = ReLU()
		# initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=output_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
		# return the output predictions
        return output

def main():
    img = mpimg.imread('../data/ctw-test-01-of-07/0000001.jpg')
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((1492.8684112354124, 1020.9181839073769), 23.764909585513124, 32.58313861351917, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    # dataset = load_dataset()
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # model = OCRNetwork(input_size = 1, output_size = 5)
    # opt = Adam(model.parameters(), lr=0.005)
    # # lossFn = torch.nn.NLLLoss()
    # # print(labels)

    # num_epochs = 10

    # accuracy = 0
    # epochs = 0
    # accuracy_arr = []
    # loss_arr = []
    # for _ in range(num_epochs):
    #     total_loss = 0
    #     correct = 0
    #     total_y = 0
        
    #     for train_x, train_y in dataloader:
    #         print(train_x.shape)
    #         print(train_y.shape)
    #         pred = model(train_x)
    #         loss = cross_entropy(pred, train_y)
    #         total_loss += loss.item()
    #         correct += torch.sum(torch.argmax(pred, dim=1) == train_y).item()
    #         total_y += len(train_y)

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
        
    #     epochs += 1
    #     accuracy = correct/total_y
    #     print("accuracy: " + str(accuracy))
    #     accuracy_arr.append(accuracy)
    #     loss_arr.append(total_loss/len(dataloader))
    
    # print(accuracy_arr)
    # print(loss_arr)



if __name__ == "__main__":
    main()
