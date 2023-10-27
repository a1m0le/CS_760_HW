import hyperparams as hp
import numpy as np
import torch
import torchvision
import math
import sys

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
        # set weights to 0
        self.W1 = torch.zeros(hp.D1, hp.D)
        self.W2 = torch.zeros(hp.K, hp.D1)
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
        self.soft = torch.nn.Softmax()


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





def grassyTRAIN(training_data, batchlimit=None, verbose=False):
    # train using my derived gradient updates.
    myNN = grassyNN()
    if batchlimit is not None and batchlimit == 0:
        return myNN
    for epoch in range(1, hp.EPOCH+1):
        batch_count = 0
        for batchdata, labels in training_data:
            myNN.batched_sgd(batchdata, labels)
            batch_count += 1
            if batchlimit is not None and batch_count == batchlimit:
                break
        if not verbose:
            continue
        #end of epoch accuracy run
        correct_ones = 0
        total_loss = 0
        avg_factor = 0
        for batchdata, labels in training_data:
            for i in range(0, len(batchdata)):
                x = torch.flatten(batchdata[i])
                y = labels[i].item()
                c,l = myNN.forwardpass(x, y, predict=True)
                if c:
                    correct_ones += 1
                total_loss += l
                avg_factor += 1
        print("Epoch "+str(epoch)+": accuracy="+str(correct_ones/avg_factor)+"\t loss="+str(total_loss/avg_factor))
    return myNN       


def evaluate_grassy(testing_data, myNN):
    correct_ones = 0
    total_loss = 0
    avg_factor = 0
    for batchdata, labels in testing_data:
        for i in range(0, len(batchdata)):
            x = torch.flatten(batchdata[i])
            y = labels[i].item()
            c, l = myNN.forwardpass(x,y, predict=True)
            if c:
                correct_ones += 1
            total_loss += l
            avg_factor += 1
    return correct_ones/avg_factor, total_loss/avg_factor



if __name__=="__main__":
    torch.set_default_dtype(torch.float64)
    train_dl = load_training_data()
    test_dl = load_testing_data()
    # train with different number of batches
    x_axis = []
    accu_axis = []
    finaltest = False
    starting = 0
    if len(sys.argv) == 2:
        if sys.argv[1] == "test":
            finaltest = True
            starting = hp.AXIS_COUNT
            print("Enter final test mode")
    for multiplier in range(starting, hp.AXIS_COUNT+1):
        myNN = grassyTRAIN(train_dl, batchlimit=multiplier*hp.GAP, verbose=finaltest) 
        accu, loss = evaluate_grassy(test_dl, myNN)
        x_axis.append(multiplier * hp.GAP * hp.BATCHSIZE)
        accu_axis.append(accu)
        precent_text = "{:.2f}%".format(multiplier*hp.GAP/len(train_dl)*100)
        accu_text = "{:.2f}%".format(accu*100)
        err_text = "{:.2f}%".format((1-accu)*100)
        print("Train with "+precent_text+" of data:  Accuracy = "+accu_text+"    Error rate = "+err_text+"   Avg.loss = "+str(loss))
    # draw the curve
    print("=========  Curve data  ===========")
    print(x_axis)
    print(accu_axis)
    if finaltest:
        exit()
    fig, axes = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    axes.set_xlim(0,60000)
    axes.set_ylim(0,1)
    axes.grid()
    axes.set_xlabel("Number of data points used in the training")
    axes.set_ylabel("Accuracy(%)")
    axes.set_title("Accuracy on the test set when changing the size of training set\nModel = my grassyNN with 0 as all initial weights\n(learning rate = "+str(hp.LEARNING_RATE)+"; batch size = "+str(hp.BATCHSIZE)+"; trained for "+str(hp.EPOCH)+" epochs each time)")
    axes.plot(x_axis, accu_axis, linestyle='-', marker='*')
    plt.savefig("Q4_4_zero_learning_cruve.png")
