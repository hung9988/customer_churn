import numpy as np
import pandas as pd 


data_pd = pd.read_csv('train.csv')

#convert the data to np array and split the set

data = np.array(data_pd)
# np.random.shuffle(data)
train_split=0.7
valid_split=0.1
test_split=0.2
X_train= data[:int(train_split*data.shape[0]),:-1]
Y_train= data[:int(train_split*data.shape[0]),-1]
X_valid= data[int(train_split*data.shape[0]):int((train_split+valid_split)*data.shape[0]),:-1]
Y_valid= data[int(train_split*data.shape[0]):int((train_split+valid_split)*data.shape[0]),-1]
X_test= data[int(-(test_split*data.shape[0])):,:-1]
Y_test= data[int(-(test_split*data.shape[0])):,-1]


#normalize the data
def one_hot(X):
    features_list=[]
    m,n=X.shape
    X_new = np.empty((m,0))
    index=0
    for i in X[0,:]:
        if isinstance(i,str):
            uniques, ids= np.unique(X[:,index], return_inverse=True)
            one_hot_temp = np.zeros((m, len(uniques)))
            one_hot_temp[np.arange(m), ids] = 1
            features_list.append(one_hot_temp)
            X_new=np.concatenate((X_new,one_hot_temp),axis=1)
        else:
            features_list.append(X[:,index].reshape(-1,1))
            X_new= np.concatenate((X_new,X[:,index].reshape(-1,1)),axis=1)
        index+=1
    return X_new,features_list

def normalize(X, features_list):
    for i in features_list:
        mean = np.mean(X[:,i])
        min = np.min(X[:,i])
        max = np.max(X[:,i])
        X[:,i] = (X[:,i]-min)/(max-min)
        
    return X

X_train = normalize(X_train, [1,np.arange(5,16,1)])
X_valid = normalize(X_valid, [1,np.arange(5,16,1)])
X_test = normalize(X_test, [1,np.arange(5,16,1)])

X_train,train_features = one_hot(X_train)
X_valid,valid_features = one_hot(X_valid)
X_test,test_features = one_hot(X_test)


#one hot encode Y set
Y_test[Y_test=='no']=0
Y_test[Y_test=='yes']=1
Y_train[Y_train=='no']=0
Y_train[Y_train=='yes']=1
Y_valid[Y_valid=='no']=0
Y_valid[Y_valid=='yes']=1
import pickle


with open('data_train.pkl', 'wb') as f:
    pickle.dump([train_features,Y_train,valid_features,Y_valid], f)
    
with open('data_test.pkl', 'wb') as f:
    pickle.dump([X_test,Y_test], f)
    
    
    




