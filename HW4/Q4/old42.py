import hyperparams as hp
import numpy as np
import torch
import torchvision
import math
import sys

import matplotlib.pyplot as plt


def load_training_data():
    # load MNIST training data
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))#TODO: Do we need to normalize
        ])
    full_dataset = torchvision.datasets.MNIST("./training_data", train=True, download=True, transform=transform_func)
    dl = torch.utils.data.DataLoader(full_dataset,batch_size=hp.BATCHSIZE,shuffle=True)
    return dl

def load_testing_data():
    # load MNIST testing data
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #TODO: Do we need to normalize
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    full_dataset = torchvision.datasets.MNIST("./testing_data", train=False, download=True, transform=transform_func)
    dl = torch.utils.data.DataLoader(full_dataset,batch_size=hp.BATCHSIZE)
    return dl


class grassyNN:

    def __init__(self, randomweight=False):
        # weights
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


    def forwardpass(self, x, y):
        assert x.shape[0] == hp.D
        # towards hidden layer
        h_raw = self.W1 @ x
        assert h_raw.shape[0] == hp.D1
        # activation
        sigma = torch.nn.Sigmoid()
        self.h = sigma(h_raw)
        assert self.h.shape[0] == hp.D1
        # towards output layer
        self.O = self.W2 @ self.h
        assert self.O.shape[0] == hp.K
        # do the softmax
        soft = torch.nn.Softmax(dim=0)
        self.sm = soft(self.O)
        assert self.sm.shape[0] == hp.K
        # we are techniquely done here but we can still compute the loss and prediction
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
        assert x.shape[0] == hp.D
        y_vec = torch.zeros(hp.K)
        y_vec[y] = 1 # construct the one hot vector.
        # gradient for W2
        sm_minus_y = self.sm - y_vec
        self.dW2 = torch.outer(sm_minus_y, self.h)
        assert self.dW2.shape[0] == hp.K and self.dW2.shape[1] == hp.D1
        # gradient for h
        self.gh = sm_minus_y @ self.W2
        assert self.gh.shape[0] == hp.D1
        # gradient for W1
        combined_h = self.gh * self.h * (1 - self.h)
        self.dW1 = torch.outer(combined_h, x)
        assert self.dW1.shape[0] == hp.D1 and self.dW1.shape[1] == hp.D
        # add it to the total gradients for this batch
        self.batch_dW1 = self.batch_dW1 + self.dW1
        self.batch_dW2 = self.batch_dW2 + self.dW2


    def batched_sgd(self, batchdata, labels):
        # zero the gradients
        self.batch_dW1 = torch.zeros(hp.D1, hp.D)
        self.batch_dW2 = torch.zeros(hp.K, hp.D1)
        gotcha_count = 0
        total_loss = 0
        # iterate
        assert len(batchdata) == len(labels)
        for i in range(0, len(batchdata)):
            # process each input and label
            x = torch.flatten(batchdata[i])
            y = labels[i].item()
            # do the gradeint
            c,l = self.forwardpass(x, y)
            if c:
                gotcha_count += 1
            total_loss += l
            self.backwardpass(x, y)
        # Now we are done, we update the weights
        avg_factor = len(batchdata)
        avg_dW1 = self.batch_dW1 / avg_factor
        avg_dW2 = self.batch_dW2 / avg_factor
        self.W1 = self.W1 - hp.LEARNING_RATE * avg_dW1
        self.W2 = self.W2 - hp.LEARNING_RATE * avg_dW2
        return gotcha_count/avg_factor, total_loss/avg_factor




def grassyTRAIN(training_data, batchlimit=None, verbose=False):
    # train using my derived gradient updates.
    myNN = grassyNN()
    print(batchlimit)
    if batchlimit is not None and batchlimit == 0:
        return myNN
    for epoch in range(1, hp.EPOCH+1):
        batch_count = 0
        for batchdata, labels in training_data:
            accu, loss = myNN.batched_sgd(batchdata, labels)
            #print("Epoch "+str(epoch)+" batch "+str(batch_count)+" accuracy="+str(accu)+"\t loss="+str(loss)+" for this batch")
            batch_count += 1
            if batchlimit is not None and batch_count == batchlimit:
                break
        if not verbose:
            continue
        #accuracy run
        correct_ones = 0
        total_loss = 0
        avg_factor = 0
        for batchdata, labels in training_data:
            for i in range(0, len(batchdata)):
                x = torch.flatten(batchdata[i])
                y = labels[i].item()
                c,l = myNN.forwardpass(x, y)
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
            c, l = myNN.forwardpass(x,y)
            if c:
                correct_ones += 1
            total_loss += l
            avg_factor += 1
    #print("=====================================")
    #print("Testing Run: accuracy="+str(correct_ones/avg_factor)+"\t loss="+str(total_loss/avg_factor))
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
    if finaltest:
        exit()
    fig, axes = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(16)
    axes.set_xlim(0,60000)
    axes.set_ylim(0,1)
    axes.plot(x_axis, accu_axis, linestyle='-', marker='*')
    plt.savefig("Q4_2_learning_cruve.png")

