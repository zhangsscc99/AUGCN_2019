import pandas as pd
import torch
def data_AU(path):
    data = pd.read_csv(path)
    
    # converting column data to list
    #AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
    
    AU1 = data['AU01_c'].tolist()
    AU2 = data['AU02_c'].tolist()
    AU4 = data['AU04_c'].tolist()
    AU5 = data['AU05_c'].tolist()
    AU6 = data['AU06_c'].tolist()
    AU7 = data['AU07_c'].tolist()
    AU9 = data['AU09_c'].tolist()
    AU10 = data['AU10_c'].tolist()
    AU12 = data['AU12_c'].tolist()
    AU14 = data['AU14_c'].tolist()
    AU15 = data['AU15_c'].tolist()
    AU17 = data['AU17_c'].tolist()
    AU20 = data['AU20_c'].tolist()
    AU23 = data['AU23_c'].tolist()
    AU25 = data['AU25_c'].tolist()
    AU26 = data['AU26_c'].tolist()
    AU28 = data['AU28_c'].tolist()
    AU45 = data['AU45_c'].tolist()
    AU_lst = [AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU28, AU45]


    lst = []
    for i in range(18):
        # print(int(AU_lst[i][0]))
        lst.append(int(AU_lst[i][0]))

    global AU_set_lst
    AU_set_lst = []
    for i in range(len(AU_lst[0])):
        set = []
        for j in range(len(AU_lst)):
            set.append(AU_lst[j][i])
        AU_set_lst.append(set)
    
    AU_set_lst2 = []
    
    for j in range(len(AU_set_lst[0])):
        ave_AU = 0
        
        for i in range(len(AU_set_lst)):
        
            ave_AU += AU_set_lst[i][j]
        
        ave_AU = ave_AU/len(AU_set_lst)
        AU_set_lst2.append(ave_AU)

    AU_set_lst2 = torch.FloatTensor(AU_set_lst2)
    AU_set_lst2 = torch.nan_to_num(AU_set_lst2, nan=0.0)
    AU_set_lst2 = AU_set_lst2.tolist()
    AU_set_lst2 = torch.LongTensor(AU_set_lst2)
    embedding = torch.nn.Embedding(num_embeddings = 18, embedding_dim = 40)
    features = embedding(AU_set_lst2)
    # print(type(features_lst))
    # print(len(AU_set_lst))
    # print(multiply_adj)
    return features, AU_set_lst
"""
def data_AU(path):
    data = pd.read_csv(path)
    
    # converting column data to list
    #AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
    
    AU1 = data['AU01_c'].tolist()
    AU2 = data['AU02_c'].tolist()
    AU4 = data['AU04_c'].tolist()
    AU5 = data['AU05_c'].tolist()
    AU6 = data['AU06_c'].tolist()
    AU7 = data['AU07_c'].tolist()
    AU9 = data['AU09_c'].tolist()
    AU10 = data['AU10_c'].tolist()
    AU12 = data['AU12_c'].tolist()
    AU14 = data['AU14_c'].tolist()
    AU15 = data['AU15_c'].tolist()
    AU17 = data['AU17_c'].tolist()
    AU20 = data['AU20_c'].tolist()
    AU23 = data['AU23_c'].tolist()
    AU25 = data['AU25_c'].tolist()
    AU26 = data['AU26_c'].tolist()
    AU28 = data['AU28_c'].tolist()
    AU45 = data['AU45_c'].tolist()
    AU_lst = [AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45]


    lst = []
    for i in range(17):
        # print(int(AU_lst[i][0]))
        lst.append(int(AU_lst[i][0]))

    global AU_set_lst
    AU_set_lst = []
    for i in range(len(AU_lst[0])):
        set = []
        for j in range(len(AU_lst)):
            set.append(AU_lst[j][i])
        AU_set_lst.append(set)
    
    AU_set_lst2 = []
    
    for j in range(len(AU_set_lst[0])):
        ave_AU = 0
        
        for i in range(len(AU_set_lst)):
        
            ave_AU += AU_set_lst[i][j]
        
        ave_AU = ave_AU/len(AU_set_lst)
        AU_set_lst2.append(ave_AU)

    AU_set_lst2 = torch.FloatTensor(AU_set_lst2)
    AU_set_lst2 = torch.nan_to_num(AU_set_lst2, nan=0.0)
    AU_set_lst2 = AU_set_lst2.tolist()
    AU_set_lst2 = torch.LongTensor(AU_set_lst2)
    embedding = torch.nn.Embedding(num_embeddings = 17, embedding_dim = 40)
    features = embedding(AU_set_lst2)
    # print(type(features_lst))
    # print(len(AU_set_lst))
    # print(multiply_adj)
    return features, AU_set_lst
"""
def data_AU_continuous(path):
    data = pd.read_csv(path)
    
    # converting column data to list
    #AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
    
    AU1 = data['AU01_r'].tolist()
    AU2 = data['AU02_r'].tolist()
    AU4 = data['AU04_r'].tolist()
    AU5 = data['AU05_r'].tolist()
    AU6 = data['AU06_r'].tolist()
    AU7 = data['AU07_r'].tolist()
    AU9 = data['AU09_r'].tolist()
    AU10 = data['AU10_r'].tolist()
    AU12 = data['AU12_r'].tolist()
    AU14 = data['AU14_r'].tolist()
    AU15 = data['AU15_r'].tolist()
    AU17 = data['AU17_r'].tolist()
    AU20 = data['AU20_r'].tolist()
    AU23 = data['AU23_r'].tolist()
    AU25 = data['AU25_r'].tolist()
    AU26 = data['AU26_r'].tolist()
    # AU28 = data['AU28_r'].tolist()
    AU45 = data['AU45_r'].tolist()
    AU_lst = [AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45]


    lst = []
    for i in range(17):
        # print(int(AU_lst[i][0]))
        lst.append(int(AU_lst[i][0]))

    global AU_set_lst
    AU_set_lst = []
    for i in range(len(AU_lst[0])):
        set=[]
        for j in range(len(AU_lst)):
            set.append(AU_lst[j][i])
        AU_set_lst.append(set)
    
    AU_set_lst2 = []
    for j in range(len(AU_set_lst[0])):
        ave_AU = 0
        for i in range(len(AU_set_lst)):
            
            ave_AU += AU_set_lst[i][j]
        ave_AU = ave_AU/len(AU_set_lst)
        
        AU_set_lst2.append(ave_AU)
        
            # print(ave_AU)
    AU_set_lst2 = torch.FloatTensor(AU_set_lst2)
    AU_set_lst2 = torch.nan_to_num(AU_set_lst2, nan=0.0)
    AU_set_lst2 = AU_set_lst2.tolist()
    AU_set_lst2 = torch.LongTensor(AU_set_lst2)
  

    # AU_set_lst = torch.FloatTensor(AU_set_lst)
    # AU_set_lst = AU_set_lst.to(torch.float32)

    embedding = torch.nn.Embedding(num_embeddings=17, embedding_dim=40)
    # embedding = torch.nn.Linear(18, 40)
    



    
    # features=embedding(AU_set_lst[0])
    # len(AU_set_lst)==1050  1320
    # print(len(AU_set_lst))
    # features=embedding(AU_set_lst[0])

    features = embedding(AU_set_lst2)
    
    # print(features.shape)
    # features = features.tolist()
    # print(type(features))-> list two ]]
    # print(features)-> 
    return features, AU_set_lst

    """

    def average(AU_set_lst):
        AU_set_lst2 = []
        # print(AU_set_lst[0])
        # print(AU_set_lst)
        for j in range(len(AU_set_lst[0])):
            ave_AU = 0
            
            for i in range(len(AU_set_lst)):
            
                ave_AU += AU_set_lst[i][j]
            
            ave_AU = ave_AU/len(AU_set_lst)
            AU_set_lst2.append(ave_AU)

        AU_set_lst2 = torch.FloatTensor(AU_set_lst2)
        AU_set_lst2 = torch.nan_to_num(AU_set_lst2, nan=0.0)
        AU_set_lst2 = AU_set_lst2.tolist()
        AU_set_lst2 = torch.LongTensor(AU_set_lst2)
        embedding = torch.nn.Embedding(num_embeddings = 18, embedding_dim = 40)
        features = embedding(AU_set_lst2)
        return features

    
    

    for i in range(len(AU_set_lst)//50):
        lst = AU_set_lst[i*50:(i+1)*50]
        # multiply_adj.append(50)
        # print(lst)

        features_lst.append(average(lst))
    if len(AU_set_lst)%50 != 0:
        inc = len(AU_set_lst) - len(AU_set_lst)%50
        last_lst = AU_set_lst[inc:]
        features_lst.append(average(last_lst))
        # multiply_adj.append(len(AU_set_lst)%50)
    if len(AU_set_lst) % 50 ==0:
        multipli = len(AU_set_lst)//50
    else:
        multipli = len(AU_set_lst)//50 + 1
"""