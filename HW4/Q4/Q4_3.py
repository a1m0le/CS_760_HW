import hyperparams as hp
import numpy as np
import torch
import torchvision
import math
import torch.nn as nn

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



def torchyEVAL(testing_data, torchNN):
    CELoss = nn.CrossEntropyLoss()
    torchNN.eval()
    with torch.no_grad():
        correct_ones = 0
        total_loss = 0
        avg_factor = 0
        for batchdata, labels in testing_data:
            forward_out = torchNN(batchdata)
            loss = CELoss(forward_out, labels)
            total_loss += loss * len(batchdata)
            avg_factor += len(batchdata)
            # accuracy
            for i in range(0, len(forward_out)):
                maxprob = max(forward_out[i]).item() # softmax not necessary here for just accuracy.
                truthprob = forward_out[i][labels[i].item()].item()
                uniform = True
                for j in range(0, len(forward_out[i])-1):
                    if forward_out[i][j] != forward_out[i][j+1]:
                        uniform = False
                if maxprob == truthprob and not uniform:
                    correct_ones += 1
    accu = correct_ones / avg_factor
    loss = total_loss / avg_factor
    accu_text = "{:.2f}%".format(accu*100)
    loss_text = "{:.4f}".format(loss)
    print("Accuracy= "+accu_text+" Loss = "+loss_text,end="")
    return accu


def torchyTRAIN(training_data, testing_data):
    # train using the power of pytorch
    torchNN = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,300,bias=False),
            nn.Sigmoid(),
            nn.Linear(300,10,bias=False),
    )
    pytorchSGD = torch.optim.SGD(torchNN.parameters(), lr=hp.LEARNING_RATE)
    CELoss = nn.CrossEntropyLoss() 
    x_axis = []
    test_accu_axis = []
    print("Before training: Test ",end="")
    first_test_accu = torchyEVAL(testing_data, torchNN)
    x_axis.append(0)
    test_accu_axis.append(first_test_accu)
    print()
    # Now we train
    for epoch in range(1, hp.EPOCH+1):
        torchNN.train()
        for batchdata, labels in training_data:
            pytorchSGD.zero_grad()
            forward_out = torchNN(batchdata)
            loss= CELoss(forward_out, labels)
            loss.backward()
            pytorchSGD.step()
        # complete one epoch. evaluate
        print("Epoch "+str(epoch)+": Test ",end="")
        test_accu = torchyEVAL(testing_data, torchNN)
        x_axis.append(epoch)
        test_accu_axis.append(test_accu)
        print()
    return x_axis, test_accu_axis



if __name__=="__main__":
    train_dl = load_training_data()
    test_dl = load_testing_data()
    x_axis, test_accu_axis = torchyTRAIN(train_dl, test_dl)
    # draw the curve
    print("=========  Curve data  ===========")
    print(x_axis)
    print(test_accu_axis)
    # Time to plot
    fig, axes = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    axes.set_xlim(0,36)
    axes.set_ylim(0,1)
    axes.grid()
    axes.set_xlabel("Epoch number")
    axes.set_ylabel("Accuracy(%)")
    axes.set_title("Accuracy on the test set as we train the model\nModel = torchNN with powers of PyTorch\n(learning rate = "+str(hp.LEARNING_RATE)+"; batch size = "+str(hp.BATCHSIZE)+")")
    axes.plot(x_axis, test_accu_axis, linestyle='-', marker='*')
    plt.savefig("Q4_3_learning_cruve.png")
