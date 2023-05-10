from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import csv
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import copy

from pygcn.utils import load_data
from pygcn.models import GCN
from pygcn.feature_matrix import *
from pygcn.adj_matrix import *
# from pygcn.load_labels2019 import load_labels


import neptune.new as neptune

# Create a Neptune run object
# record1
run = neptune.init_run(
    project="depressiondiagnosis/depressiondiagnosis",  
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzRkYjU2OC05ZWZkLTQ4YTUtYTA1OS04OGUyYTUxOGY4MmQifQ==",
)  # your credentials

run["algorithm"] = "AU-GCN"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

PARAMS = {
    # "batch_size": 64,
    "dropout": args.dropout,
    "learning_rate": args.lr,
    "optimizer": "Adam",
    "weight_decay": args.weight_decay,
    "hidden_layers": args.hidden
}
run["parameters"] = PARAMS

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# prepare labels
# labels = load_labels('/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind', '/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_DepressionLabels/Training_DepressionLabels')

# prepare features for training data
list_train = []

labels = []
with open("/Users/mac/Desktop/AUGCN/train_split.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        label = []
        if row[0].isdigit():
            list_train.append(row[0])
            #print(row[3])
            label.append(int(row[3]))
            label = torch.LongTensor(label)
            label = label.to(torch.float32)

            labels.append(label)
        # print(row[0])
labels = torch.stack(labels, dim=0)
            

def select_paths(paths, list_of_num):
    res_lst = []
    label = []
    for i in range(len(paths)):
        if paths[i][:3] in list_of_num:
            res_lst.append(paths[i])
            label.append(paths[i][:3])
    return res_lst, label


p_training = "/Users/mac/Desktop/AUGCN/avec2019"
paths_features = os.listdir(p_training)
paths_features = sorted(paths_features)


paths_features, label_train = select_paths(paths_features, list_train)
# print(101)

# print(paths_features)
# print(len(paths_features), len(list_train))


def add_prefix(paths, p_training):
    # paths = sorted(paths)
    # print(paths)

    
    
    for i in range(len(paths)):
        
        paths[i] = p_training + "/" + paths[i]
    return paths


paths_features = add_prefix(paths_features, p_training)
print(len(paths_features))
# print(paths_features)
# print(paths_features)
big_lst = []
for i in range(len(paths_features)): 
    features_inside = data_AU(paths_features[i])[0]
    big_lst.append(features_inside)
features = torch.stack(big_lst, dim=0)

print(len(features), len(labels))
print(102)


# prepare adj matrix


def get_all_adj(AU_set_lst):
    res=[]
    for i in range(18):
        for j in range(18):
            res.append(adj_matrix(i, j, AU_set_lst))
                   
    x = np.array(res)        # x是一维数组 
    d = x.reshape((18, 18))   # 将x重塑为2行4列的二维数组
    return d

def get_all_adj(AU_set_lst):
    res=[]
    for i in range(18):
        for j in range(18):
            res.append(adj_matrix(i, j, AU_set_lst))
                   
    x = np.array(res)        # x是一维数组 
    d = x.reshape((18, 18))   # 将x重塑为2行4列的二维数组
    return d

"""
def get_all_adj_17(AU_set_lst):
    res=[]
    for i in range(17):
        for j in range(17):
            res.append(adj_matrix(i,j,AU_set_lst))
                   
    x = np.array(res)        # x是一维数组 
    d = x.reshape((17, 17))   # 将x重塑为2行4列的二维数组
    return d
"""


# 3D dimensional adj matrix

def format_adj(paths_features):

    adj_lst = []
    for i in range(len(paths_features)):
        AU_set_lst = data_AU(paths_features[i])[-1] 
        #multipli = data_AU(paths_features[i])[-2]

        adj = get_all_adj(AU_set_lst)
        adj = torch.FloatTensor(adj)
        adj_lst.append(adj)
        #for j in range(multipli):
        #    adj_lst.append(adj)
        #    labels2.append(labels[i])
    return torch.stack(adj_lst,dim=0)




adj = format_adj(paths_features)
#labels = copy.deepcopy(labels2)
#labels = torch.stack(labels, dim=0)

# development data
# /Users/mac/Desktop/AUGCN/pygcn/Development /csvNorthwind


# labels_dev = load_labels("/Users/mac/Desktop/AUGCN/pygcn/Development/csvNorthwind", '/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_DepressionLabels/Development_DepressionLabels')

print("dev begins")

list_dev = []
labels_dev = []
with open("/Users/mac/Desktop/AUGCN/dev_split.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        label = []
        if row[0].isdigit():
            list_dev.append(row[0])
            label.append(int(row[3]))
            label = torch.LongTensor(label)
            label = label.to(torch.float32)
            labels_dev.append(label)

labels_dev = torch.stack(labels_dev, dim=0)
p_dev = "/Users/mac/Desktop/AUGCN/avec2019"
paths_features_dev = os.listdir(p_dev)
paths_features_dev = sorted(paths_features_dev)







paths_features_dev, label_dev = select_paths(paths_features_dev, list_dev)
paths_features_dev = add_prefix(paths_features_dev, p_dev)
big_lst_dev = []
for i in range(len(paths_features_dev)): 
    features_inside = data_AU(paths_features_dev[i])[0]
    big_lst_dev.append(features_inside)
features_dev = torch.stack(big_lst_dev, dim=0)
print(len(features_dev), len(labels_dev))
print(102)

adj_dev = format_adj(paths_features_dev)
#labels_dev = copy.deepcopy(labels2_dev)
#labels_dev = torch.stack(labels_dev, dim=0)


# testing data
# labels_test = load_labels('/Users/mac/Desktop/AUGCN/pygcn/Testing/csvNorthwind', '/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_Labels_Testset/Testing/DepressionLabels')


list_test = []
labels_test = []
with open("/Users/mac/Desktop/AUGCN/test_split.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        label = []
        if row[0].isdigit():
            list_test.append(row[0])
            label.append(int(row[3]))
            label = torch.LongTensor(label)
            label = label.to(torch.float32)
            labels_test.append(label)

labels_test = torch.stack(labels_test, dim=0)
print(labels_test)
p_testing = "/Users/mac/Desktop/AUGCN/avec2019"
paths_features_test = os.listdir(p_testing)
paths_features_test = sorted(paths_features_test)




paths_features_test, label_test = select_paths(paths_features_test, list_test)
paths_features_test = add_prefix(paths_features_test, p_testing)

big_lst_test = []
for i in range(len(paths_features_test)): 
    features_inside = data_AU(paths_features_test[i])[0]
    big_lst_test.append(features_inside)
features_test = torch.stack(big_lst_test, dim=0)

print(len(features_test), len(labels_test))


adj_test = format_adj(paths_features_test)
#labels_test = copy.deepcopy(labels2_test)
#labels_test = torch.stack(labels_test, dim=0)



# Model and optimizer
model = GCN(nfeat=features.shape[-1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
#optimizer = optim.Adam(model.parameters(),
#                       lr=args.lr, weight_decay=args.weight_decay)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# the loss function I set. MSE
loss_func = torch.nn.MSELoss()
loss_func_MAE = torch.nn.L1Loss()

# model training 

print("dataset description: ")
print("how many videos we have used for training: " + str(adj.shape[0]))
print("adjacency matrix dimension: ")
print(adj.shape)

print("node matrix dimension: ")
print(features.shape)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    # print(output)
    print("output dimension")
    print(output.shape)
    # print(output[idx_train],labels[idx_train])

    idx_train = []
    for i in range(0, len(output)):
        idx_train.append(i)
    
    
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = torch.sqrt(loss_func(output[idx_train], labels[idx_train]))
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train_MAE = loss_func_MAE(output[idx_train], labels[idx_train])
    loss_train.backward()
    
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        # print(output)

    output = model(features_dev, adj_dev)
    global idx_val

    idx_val = []
    for i in range(len(output)):
        idx_val.append(i)

    loss_val = torch.sqrt(loss_func(output[idx_val], labels_dev[idx_val]))
    loss_val_MAE = loss_func_MAE(output[idx_val], labels_dev[idx_val])
    # loss_val.backward()

    run["train/loss_RMSE"].append(loss_train)  
    run["train/loss_MAE"].append(loss_train_MAE)

    run["val/loss_RMSE"].append(loss_val)  
    run["val/loss_MAE"].append(loss_val_MAE)
    
    # loss_val = torch.sqrt(loss_func(output[idx_val], labels[idx_val]))
    
    print('Epoch: {:04d}'.format(epoch+1),
          'Training set results: loss_train: {:.4f}'.format(loss_train.item()),
          "(RMSE)",
          'loss_train_MAE: {:.4f}'.format(loss_train_MAE.item()),
          "(MAE)",
          '\nDevelopment set results: loss_val: {:.4f}'.format(loss_val.item()),
          "(RMSE)",
          'loss_val_MAE: {:.4f}'.format(loss_val_MAE.item()),
          "(MAE)")
          # 'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features_test, adj_test)
    # print(output.shape)
    
    idx_test = []
    for i in range(len(output)):
        idx_test.append(i)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = torch.sqrt(loss_func(output[idx_test], labels_test[idx_test]))
    # loss_val = torch.sqrt(loss_func(output[idx_val], labels_dev[idx_val]))

    loss_test_MAE = loss_func_MAE(output[idx_test], labels_test[idx_test])
    


    run["test/loss_RMSE"].append(loss_test)  
    run["test/loss_MAE"].append(loss_test_MAE)
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          # "loss= {:.4f}".format(loss_test.item()),
          "loss= {:.4f}".format(loss_test.item()),
          "(RMSE)",
          "loss_MAE= {:.4f}".format(loss_test_MAE.item()),
          "(MAE)",
          )


train(args.epochs)
test()

run.stop()

# print(features.shape)
# from torchsummary import summary
# print(summary(model))


