##DATA PREPROCESSING PART##
import numpy as np
import pandas as pd

df=pd.read_csv('train.csv')

data=np.array(df)
np.random.shuffle(data)
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
data[data=='no']=0.
data[data=='yes']=1.
X = data[:, :-1]
y = data[:, -1]
features_columns = df.columns[:-1]
X, features_dict = one_hot(X, features_columns)
y=y.astype(float)
X=X.astype(float)

features_dict['area_code'].shape

def feature_selection(features_dict, feature_names):
    X_features=np.zeros((features_dict[feature_names[0]].shape[0],0))
    for feature in feature_names:
        X_features=np.concatenate((X_features,features_dict[feature]),axis=1)
           
    return X_features
features_columns_excluding_state_and_area_code = [col for col in features_columns if col not in ['state', 'area_code', 'account_length']]
X = feature_selection(features_dict, features_columns_excluding_state_and_area_code)

def normalize(data,features):
    for i in features:
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    
    return data

X = normalize(X, np.arange(X.shape[1]))

split_train = int(0.7 * X.shape[0])
split_valid= int(0.15* X.shape[0])
split_test = int(0.15 * X.shape[0])

X_train = X[:split_train]
y_train = y[:split_train]
X_valid = X[split_train:split_train+split_valid]
y_valid = y[split_train:split_train+split_valid]
X_test = X[split_train+split_valid:]
y_test = y[split_train+split_valid:]



#### TRAINING PART ####
def init_weight_and_bias(n):
    return np.random.randn(n), np.random.randn(1)



def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(X,W,b):
   
    A=np.dot(X,W)+b
    A=A.astype(float)
    return sigmoid(A)
def loss(y,y_hat):
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def train(X, y, W, b, lr, epochs, reg):
    costs=[]
  
    for epoch in range(epochs):
        
            # Forward pass
        
            y_hat = forward(X, W, b)
            
            # Loss calculation
            l = loss(y, y_hat)
            costs.append(l)
        

            # Gradient calculation
            print('Epoch:',epoch,',Loss:',l)
            
            Z=y_hat-y
            dW = (np.dot(X.T, Z) + reg * W)/X.shape[0]
          
            db = np.mean(Z)

            # Parameter updates
            dW=dW.astype(float)
            W -= lr * dW
            b -= lr * db

    return W, b,costs

def accuracy(W,b,X,y):
    y_hat = forward(X, W, b)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    return np.mean(y_hat == y)

def fit_logistic(X, y, lr, epochs, reg):
    W_fit, b_fit = init_weight_and_bias(X.shape[1])
    
    W_fit, b_fit, costs = train(X, y, W_fit, b_fit, lr, epochs, reg)
    return W_fit, b_fit, costs


lr_value=[0.01,0.1,0.5,1,2,10]
reg_value=[0.01,0.1,0.5,1,2,10]
epochs=[1000,5000,10000,20000]
max_accuracy=0
default_epoch=15000
default_lr=0.05
default_reg=0.1
with open('output.txt', 'w') as f:
    pass
for lr in lr_value:
    
   W,b=np.random.randn(X_train.shape[1]),np.random.randn(1)
   with open('output.txt', 'a') as f:
        f.write(f"Training with epoch={default_epoch}, lr={lr}, reg={default_reg}\n")
       
        f.write("-------------------------------------------------\n")
        W, b, costs = fit_logistic(X_train, y_train, lr, default_epoch, default_reg)
        y_hat = forward(X_valid, W, b)

        f.write('\n')
        f.write(f"Validation Loss: {loss(y_valid, y_hat)}\n")
        f.write(f"Validation Accuracy: {accuracy(W, b, X_valid, y_valid)}\n")
        if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
            
        f.write("Test loss:{}\n".format(loss(y_test, forward(X_test, W, b))))
        f.write("Test accuracy:{}\n".format(accuracy(W, b, X_test, y_test)))
      
        f.write("-------------------------------------------------\n")
        f.write("\n")
    
f.close()

for reg in reg_value:
    
   W,b=np.random.randn(X_train.shape[1]),np.random.randn(1)
   with open('output.txt', 'a') as f:
        f.write(f"Training with epoch={default_epoch}, lr={default_lr}, reg={reg}\n")
        
        f.write("-------------------------------------------------\n")
        W, b, costs = fit_logistic(X_train, y_train, default_lr, default_epoch, reg)
        y_hat = forward(X_valid, W, b)

        f.write('\n')
        f.write(f"Validation Loss: {loss(y_valid, y_hat)}\n")
        f.write(f"Validation Accuracy: {accuracy(W, b, X_valid, y_valid)}\n")
        if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
            
        f.write("Test loss:{}\n".format(loss(y_test, forward(X_test, W, b))))
        f.write("Test accuracy:{}\n".format(accuracy(W, b, X_test, y_test)))
      
        f.write("-------------------------------------------------\n")
        f.write("\n")
    
    
f.close()


for epoch in epochs:
    
   W,b=np.random.randn(X_train.shape[1]),np.random.randn(1)
   with open('output.txt', 'a') as f:
        f.write(f"Training with epoch={epoch}, lr={default_lr}, reg={default_reg}\n")
      
        f.write("-------------------------------------------------\n")
        W, b, costs = fit_logistic(X_train, y_train, default_lr, epoch, default_reg)
        y_hat = forward(X_valid, W, b)

        f.write('\n')
        f.write(f"Validation Loss: {loss(y_valid, y_hat)}\n")
        f.write(f"Validation Accuracy: {accuracy(W, b, X_valid, y_valid)}\n")
        if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
            
        f.write("Test loss:{}\n".format(loss(y_test, forward(X_test, W, b))))
        f.write("Test accuracy:{}\n".format(accuracy(W, b, X_test, y_test)))
      
        f.write("-------------------------------------------------\n")
        f.write("\n")
    
f.close()

np.savez('weights.npz', W=W_best_fit, b=b_best_fit,costs=costs_best_fit)




    
    