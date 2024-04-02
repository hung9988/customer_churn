##DATA PREPROCESSING PART##
import numpy as np
import pandas as pd

df=pd.read_csv('test.csv')

data=np.array(df)

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
X = data

X, features_dict = one_hot(X, df.columns)
X=X.astype(float)



def feature_selection(features_dict, feature_names):
    X_features=np.zeros((features_dict[feature_names[0]].shape[0],0))
    for feature in feature_names:
        X_features=np.concatenate((X_features,features_dict[feature]),axis=1)
           
    return X_features
features_columns_excluding_state_and_area_code = [col for col in df.columns if col not in ['state', 'area_code', 'account_length','id']]
X = feature_selection(features_dict, features_columns_excluding_state_and_area_code)

def normalize(data,features):
    for i in features:
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    
    return data

X = normalize(X, np.arange(X.shape[1]))

X_test = X

weights=np.load('weights.npz',allow_pickle=True)

W=weights['W']
b=weights['b']

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(X,W,b):
   
    A=np.dot(X,W)+b
    A=A.astype(float)
    return sigmoid(A)


y_pred = forward(X_test,W,b)

y_pred_new=np.where(y_pred>0.5,'yes','no')


id_column = np.arange(1, y_pred_new.shape[0] + 1)

# Create a DataFrame
df_output = pd.DataFrame({
    'id': id_column,
    'churn': y_pred_new
})

# Save the DataFrame as a CSV file
df_output.to_csv('output.csv', index=False)