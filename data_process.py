
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
raw_data = pd.read_csv('train.csv')
pretty_df = raw_data.head(10).to_string(index=False)

# Print the pretty string

# Continue with your code
raw_data = np.array(raw_data)
#np data
raw_data= np.array(raw_data)

split = 0.8

train_set = raw_data[:int(split*len(raw_data))]
validation_set = raw_data[int(split*len(raw_data)):]

X_train = train_set[:,:-1]
y_train = train_set[:,-1]
X_validation = validation_set[:,:-1]
y_validation = validation_set[:,-1]

m,n=X_train.shape
X_train_new = np.empty((m,0))
index=0
for i in X_train[0,:]:
    if isinstance(i,str):
        uniques, ids= np.unique(X_train[:,index], return_inverse=True)
        one_hot_temp = np.zeros((m, len(uniques)))
        one_hot_temp[np.arange(m), ids] = 1
        X_train_new=np.concatenate((X_train_new,one_hot_temp),axis=1)
    else:
        X_train_new= np.concatenate((X_train_new,X_train[:,index].reshape(-1,1)),axis=1)
    index+=1
        

print (np.unique(X_train[:,2]))
        
   
               

