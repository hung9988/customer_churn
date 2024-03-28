#DATA PROCESSING
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
raw_data = pd.read_csv('train.csv')
pretty_df = raw_data.head(10).to_string(index=False)


raw_data = np.array(raw_data)

raw_data= np.array(raw_data)

#SPLIT THE DATA INTO TRAIN AND VALIDATION SET 

split = 0.8

train_set = raw_data[:int(split*len(raw_data))]
validation_set = raw_data[int(split*len(raw_data)):]
# SPLIT THE DATA INTO X AND Y
X_train = train_set[:,:-1]

y_train = train_set[:,-1]
X_validation = validation_set[:,:-1]
y_validation = validation_set[:,-1]

#ONE HOT ENCODING CATEGORICAL DATA

def one_hot(X_train):
    features_list=[]
    m,n=X_train.shape
    X_train_new = np.empty((m,0))
    index=0
    for i in X_train[0,:]:
        if isinstance(i,str):
            uniques, ids= np.unique(X_train[:,index], return_inverse=True)
            one_hot_temp = np.zeros((m, len(uniques)))
            one_hot_temp[np.arange(m), ids] = 1
            features_list.append(one_hot_temp)
            X_train_new=np.concatenate((X_train_new,one_hot_temp),axis=1)
        else:
            features_list.append(X_train[:,index].reshape(-1,1))
            X_train_new= np.concatenate((X_train_new,X_train[:,index].reshape(-1,1)),axis=1)
        index+=1
    return X_train_new,features_list   


X_train, features_train = one_hot(X_train)
X_validation, features_valid = one_hot(X_validation)

import pickle

with open('features_train.pkl', 'wb') as f:
    pickle.dump(features_train, f)
with open('features_valid.pkl', 'wb') as f:
    pickle.dump(features_valid, f)