import os
import csv
import torch
#164+57 +57  114   282



#paths_features = add_prefix(paths_features, p_training)

# print(paths_features)
big_lst = []
#for i in range(len(paths_features)): 
#    features_inside = data_AU(paths_features[i])[0]
#    big_lst.append(features_inside)
#features = torch.stack(big_lst, dim=0)

"""
print(len(os.listdir("/Users/mac/Desktop/AUGCN/avec2019")))
print("/Users/mac/Desktop/AUGCN/Detailed_PHQ8_Labels.csv")
print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_DepressionLabels/Training_DepressionLabels')))
print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_DepressionLabels/Development_DepressionLabels')))

print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Training/csvFreeform')))
print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind')))


print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Development/csvNorthwind')))
print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Development/csvFreeform')))


print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Testing/csvFreeform')))
print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Testing/csvNorthwind')))


list_test = []

with open("/Users/mac/Desktop/AUGCN/test_split.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        if row[0].isdigit():
            list_train.append(row[0])
p_testing = "/Users/mac/Desktop/AUGCN/avec2019"
paths_features = os.listdir(p_training)
paths_features = sorted(paths_features)


paths_features = select_paths(paths_features, list_test)
paths_features = add_prefix(paths_features, p_testing)
"""