# path1 path2
import os
import torch
def load_labels(paths_of_features, paths_of_labels):
    p1 = paths_of_features
    p2 = paths_of_labels

    paths_labels = os.listdir(p2)
    paths_labels = sorted(paths_labels)



    # Load data
    # adj, features, labels, idx_train, idx_val, idx_test = load_data()
    # reading CSV file
    # data preparation
    # node matrix-> adj matrix
    # file paths reading
    # labels paths reading
    paths_features = os.listdir(p1)
    paths_features = sorted(paths_features)
    selected_paths_labels = []
    for i in range(len(paths_features)):
        to_match = paths_features[i][:5]
        for j in range(len(paths_labels)):
            if paths_labels[j][:5] == to_match:
                selected_paths_labels.append(paths_labels[j])

    for i in range(len(selected_paths_labels)):
        selected_paths_labels[i] = p2 + "/" + selected_paths_labels[i]

    labels = []
    for i in range(len(selected_paths_labels)):
        
        with open(selected_paths_labels[i], 'r') as f:
            label = []
            label.append(int(f.readline()))
            label = torch.LongTensor(label)
            label = label.to(torch.float32)
            # for j in range(50):

            labels.append(label)

    labels = torch.stack(labels, dim=0)

    return labels