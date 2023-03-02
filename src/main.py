# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class F1(Module):
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        m = Uniform(-1/math.sqrt(h), 1/math.sqrt(h))
        layer_0 = m.sample([h, d])
        bias_0 = m.sample([h])
        self.layer_0 = Parameter(layer_0)
        self.bias_0 = Parameter(bias_0)

        n = Uniform(-1/math.sqrt(k), 1/math.sqrt(k))
        layer_1 = n.sample([k, h])
        bias_1 = n.sample([k])
        self.layer_1 = Parameter(layer_1)
        self.bias_1 = Parameter(bias_1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = torch.matmul(self.layer_0, x.T)
        for i in range(len(x[0])):
            x[:, i] += self.bias_0
        x = relu(x)
        # x is already transposed
        x = torch.matmul(self.layer_1, x)
        for i in range(len(x[0])):
            x[:, i] += self.bias_1
        return x.T


class F2(Module):
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        m = Uniform(-1/math.sqrt(h0), 1/math.sqrt(h0))
        layer_0 = m.sample([h0, d])
        bias_0 = m.sample([h0])
        self.layer_0 = Parameter(layer_0)
        self.bias_0 = Parameter(bias_0)

        m = Uniform(-1/math.sqrt(h1), 1/math.sqrt(h1))
        layer_1 = m.sample([h1, h0])
        bias_1 = m.sample([h1])
        self.layer_1 = Parameter(layer_1)
        self.bias_1 = Parameter(bias_1)

        m = Uniform(-1/math.sqrt(k), 1/math.sqrt(k))
        layer_2 = m.sample([k, h1])
        bias_2 = m.sample([k])
        self.layer_2 = Parameter(layer_2)
        self.bias_2 = Parameter(bias_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = torch.matmul(self.layer_0, x.T)
        for i in range(len(x[0])):
            x[:, i] += self.bias_0
        x = relu(x)
        # x is already transposed
        x = torch.matmul(self.layer_1, x)
        for i in range(len(x[0])):
            x[:, i] += self.bias_1
        x = relu(x)
        x = torch.matmul(self.layer_2, x)
        for i in range(len(x[0])):
            x[:, i] += self.bias_2
        return x.T


def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    accuracy = 0
    epochs = 0
    accuracy_arr = []
    loss_arr = []
    while accuracy < .99:
        total_loss = 0
        correct = 0
        total_y = 0
        
        for train_x, train_y in train_loader:
            pred = model(train_x)
            loss = cross_entropy(pred, train_y)
            total_loss += loss.item()
            correct += torch.sum(torch.argmax(pred, dim=1) == train_y).item()
            total_y += len(train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epochs += 1
        accuracy = correct/total_y
        print("accuracy: " + str(accuracy))
        accuracy_arr.append(accuracy)
        loss_arr.append(total_loss/len(train_loader))
    return loss_arr

def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    model = F1(64, 784, 10)
    # model = F2(32, 32, 784, 10)
    loss_arr = train(model, optimizer=Adam(model.parameters(), lr=0.005), train_loader=DataLoader(TensorDataset(x, y), batch_size=64))
    
    x = [i for i in range(len(loss_arr))]
    plt.plot(x, loss_arr, label = "loss")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.title("average loss vs epoch")
    plt.show()

    test_pred = model(x_test)
    loss = cross_entropy(test_pred, y_test)
    accuracy = torch.sum(torch.argmax(test_pred, dim=1) == y_test).item()/len(y_test)
    print("test loss: " + str(loss) + " accuracy: " + str(accuracy))



if __name__ == "__main__":
    main()
