##DATA PREPROCESSING PART##
import numpy as np
import pandas as pd


def one_hot(X,column_names):
    features_dict = {}
    m,n=X.shape
    X_new = np.empty((m,0))
    index=0
    for i in X[0,:]:
        if isinstance(i,str):
            uniques, ids= np.unique(X[:,index], return_inverse=True)
            one_hot_temp = np.zeros((m, len(uniques)))
            one_hot_temp[np.arange(m), ids] = 1
            features_dict[column_names[index]] = one_hot_temp
            X_new=np.concatenate((X_new,one_hot_temp),axis=1)
        else:
            features_dict[column_names[index]] = X[:, index].reshape(-1, 1)
            X_new= np.concatenate((X_new,X[:,index].reshape(-1,1)),axis=1)
        index+=1
        
    return X_new,features_dict

def normalize(data,features):
    for i in features:
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    return data


def feature_selection(features_dict, feature_names):
    X_features=np.zeros((features_dict[feature_names[0]].shape[0],0))
    for feature in feature_names:
        X_features=np.concatenate((X_features,features_dict[feature]),axis=1)
           
    return X_features


def data_process(data,train_split,valid_split,test_split,features_name, features_selection_list):
    data[data=='no']=0.
    data[data=='yes']=1.
    
    np.random.shuffle(data)
   
    X = data[:, :-1]
    y = data[:, -1]
    
    X, features_dict = one_hot(X, features_name)

    X = feature_selection(features_dict, features_selection_list)
    X = normalize(X, np.arange(X.shape[1]))
   
    split_train = int(train_split * X.shape[0])
    split_valid= int(valid_split* X.shape[0])
    split_test = int(test_split * X.shape[0])
    X_train = X[:split_train]
    y_train = y[:split_train]
    X_valid = X[split_train:split_train+split_valid]
    y_valid = y[split_train:split_train+split_valid]
    X_test = X[split_train+split_valid:]
    y_test = y[split_train+split_valid:]

    return X_train,y_train,X_valid,y_valid,X_test,y_test
