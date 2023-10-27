import hyperparams as hp
import numpy as np
import torch
import torchvision
import math

import matplotlib.pyplot as plt


def load_training_data():
    # load MNIST training data
    """
         CITATION: Transform and normalization parameters are learned from examples from pytorch: https://github.com/pytorch/examples/blob/main/mnist/main.py
    """
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    full_dataset = torchvision.datasets.MNIST("./training_data", train=True, download=True, transform=transform_func)
    dl = torch.utils.data.DataLoader(full_dataset,batch_size=hp.BATCHSIZE)
    return dl

def load_testing_data():
    # load MNIST testing data
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    full_dataset = torchvision.datasets.MNIST("./testing_data", train=False, download=True, transform=transform_func)
    dl = torch.utils.data.DataLoader(full_dataset,batch_size=hp.BATCHSIZE)
    return dl


class grassyNN:

    def __init__(self):
        # set weights
        self.W1 = torch.rand(hp.D1, hp.D) * 2 - 1
        self.W2 = torch.rand(hp.K, hp.D1) * 2 - 1
        # layers
        self.h = torch.zeros(hp.D1, 1)
        self.O = torch.zeros(hp.K, 1)
        self.sm = torch.zeros(hp.K, 1)
        # single pass gradients
        self.dW1 = torch.zeros(hp.D1, hp.D)
        self.dW2 = torch.zeros(hp.K, hp.D1)
        self.gh = torch.zeros(hp.D1, 1)
        # batch gradients
        self.batch_dW1 = torch.zeros(hp.D1, hp.D)
        self.batch_dW2 = torch.zeros(hp.K, hp.D1)
        # activation functions
        self.sigma = torch.nn.Sigmoid()
        self.soft = torch.nn.Softmax(dim=0)


    def forwardpass(self, x, y, predict=False):
        # towards hidden layer
        h_raw = self.W1 @ x
        # activation
        self.h = self.sigma(h_raw)
        # towards output layer
        self.O = self.W2 @ self.h
        # do the softmax
        self.sm = self.soft(self.O)
        if not predict:
            return
        loss = -1 * math.log(self.sm[y].item()) # only the y_th iterm is 1 and the rest are zero.
        # a cheap way to check for correct prediction
        prob_of_truth = self.sm[y].item()
        max_prob = max(self.sm).item()
        correct = max_prob == prob_of_truth # It is correct if the truth prob is the same as the max_prob and not 0
        is_uniform = True
        for i in range(0,len(self.sm)-1):
            if self.sm[i] != self.sm[i+1]:
                is_uniform = False
        return correct and not is_uniform, loss # If it is uniform, we cannot make a correct prediction at all.




    def backwardpass(self, x, y):
        y_vec = torch.zeros(hp.K)
        y_vec[y] = 1 # construct the one hot vector.
        # gradient for W2
        sm_minus_y = self.sm - y_vec
        self.dW2 = torch.outer(sm_minus_y, self.h)
        # gradient for h
        self.gh = sm_minus_y @ self.W2
        # gradient for W1
        combined_h = self.gh * self.h * (1 - self.h)
        self.dW1 = torch.outer(combined_h, x)
        # add it to the total gradients for this batch
        self.batch_dW1 = self.batch_dW1 + self.dW1
        self.batch_dW2 = self.batch_dW2 + self.dW2


    def batched_sgd(self, batchdata, labels):
        # zero the gradients
        self.batch_dW1 = torch.zeros(hp.D1, hp.D)
        self.batch_dW2 = torch.zeros(hp.K, hp.D1)
        # iterate
        for i in range(0, len(batchdata)):
            # process each input and label
            x = torch.flatten(batchdata[i])
            y = labels[i].item()
            # do the gradeint
            self.forwardpass(x, y)
            self.backwardpass(x, y)
        # Now we are done, we update the weights
        avg_factor = len(batchdata)
        avg_dW1 = self.batch_dW1 / avg_factor
        avg_dW2 = self.batch_dW2 / avg_factor
        self.W1 = self.W1 - hp.LEARNING_RATE * avg_dW1
        self.W2 = self.W2 - hp.LEARNING_RATE * avg_dW2




def grassyEVAL(data, myNN):
    correct_ones = 0
    total_loss = 0
    avg_factor = 0
    for batchdata, labels in data:
        for i in range(0, len(batchdata)):
            x = torch.flatten(batchdata[i])
            y = labels[i].item()
            c, l = myNN.forwardpass(x, y, predict=True)
            if c:
                correct_ones += 1
            total_loss += l
            avg_factor += 1
    accu = correct_ones / avg_factor
    loss = total_loss / avg_factor
    accu_text = "{:.2f}%".format(accu*100)
    loss_text = "{:.4f}".format(loss)
    print("Accuracy= "+accu_text+" Loss = "+loss_text,end="")
    return accu




def grassyTRAIN(training_data, test_data):
    # train using my derived gradient updates.
    myNN = grassyNN()
    x_axis = []
    test_accu_axis = []
    print("Before training: Test ",end="")
    first_test_accu = grassyEVAL(test_data, myNN)
    x_axis.append(0)
    test_accu_axis.append(first_test_accu)
    print()
    # Now we start training
    for epoch in range(1, hp.EPOCH+1):
        for batchdata, labels in training_data:
            myNN.batched_sgd(batchdata, labels)
        #end of epoch accuracy run
        print("Epoch "+str(epoch)+": Test ",end="")
        test_accu = grassyEVAL(test_data, myNN)
        x_axis.append(epoch)
        test_accu_axis.append(test_accu)
        print()
    return x_axis, test_accu_axis 




if __name__=="__main__":
    train_dl = load_training_data()
    test_dl = load_testing_data()
    x_axis, test_accu_axis = grassyTRAIN(train_dl, test_dl) 
    # draw the curve
    print("=========  Curve data  ===========")
    print(x_axis)
    print(test_accu_axis)
    # Now we plot
    fig, axes = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    axes.set_xlim(0,36)
    axes.set_ylim(0,1)
    axes.grid()
    axes.set_xlabel("Epoch number")
    axes.set_ylabel("Accuracy(%)")
    axes.set_title("Accuracy on the test as we train the model \nModel = my grassyNN with random initial weights\n(learning rate = "+str(hp.LEARNING_RATE)+"; batch size = "+str(hp.BATCHSIZE))
    axes.plot(x_axis, test_accu_axis, linestyle='-', marker='*')
    plt.savefig("Q4_4_rand_learning_cruve.png")
