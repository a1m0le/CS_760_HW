import hyperparams as hp
import numpy as np
import torch
import torchvision
import math
import torch.nn as nn


def load_training_data():
    # load MNIST training data
    transform_func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))#TODO: Do we need to normalize
        ])
    full_dataset = torchvision.datasets.MNIST("./training_data", train=True, download=True, transform=transform_func)
    dl = torch.utils.data.DataLoader(full_dataset,batch_size=hp.BATCHSIZE)
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






def torchyTRAIN(training_data):
    # train using my derived gradient updates.
    torchNN = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,300,bias=False),
            nn.Sigmoid(),
            nn.Linear(300,10,bias=False),
    )
    pytorchSGD = torch.optim.SGD(torchNN.parameters(), lr=hp.LEARNING_RATE)
    CELoss = nn.CrossEntropyLoss() 
    torchNN.train()
    for epoch in range(1, hp.EPOCH+1):
        total_loss = 0
        for batchdata, labels in training_data:
            pytorchSGD.zero_grad()
            forward_out = torchNN(batchdata)
            loss= CELoss(forward_out, labels)
            loss.backward()
            pytorchSGD.step()
        # get accuracy
        correct_ones = 0
        total_loss = 0
        avg_factor = 0
        for batchdata, labels in training_data:
            forward_out = torchNN(batchdata)
            loss = CELoss(forward_out, labels)
            total_loss += loss * len(batchdata)
            avg_factor += len(batchdata)
            # accuracy
            for i in range(0, len(forward_out)):
                maxprob = max(forward_out[i]).item() # softmax not necessary here for just accuracy.
                truthprob =forward_out[i][labels[i].item()].item()
                if maxprob == truthprob:
                    correct_ones += 1
        print("Epoch "+str(epoch)+": accuracy="+str(correct_ones/avg_factor)+"\t loss="+str((total_loss/avg_factor).item()))
    return torchNN       


def evaluate_torchy(testing_data, torchNN):
    correct_ones = 0
    total_loss = 0
    avg_factor = 0
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
            for i in range(0, len(batchdata)):
                maxprob = max(batchdata[i]).item() # softmax not necessary here for just accuracy.
                truthprob = batchdata[i][labels[i].item()].item()
                if maxprob == truthprob:
                    correct_ones += 1
    print("=====================================")
    print("Testing Run: accuracy="+str(correct_ones/avg_factor)+"\t loss="+str(total_loss/avg_factor))





if __name__=="__main__":
    train_dl = load_training_data()
    torchNN = torchyTRAIN(train_dl) 
    test_dl = load_testing_data()
    evaluate_grassy(test_dl, torchNN)
