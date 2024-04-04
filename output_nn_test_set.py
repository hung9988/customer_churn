from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


df =pd.read_csv('test.csv')

### UNCOMMENT THIS PART TO USE THE FEATURE ENGINEERING
df['total_call'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']

# Create 'total_charges' feature
df['total_charges'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']

# Create 'total_minutes' feature
df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
df = df.drop(['total_day_calls', 'total_eve_calls', 'total_night_calls'], axis=1)

# Delete contributing features for 'total_charges'
df = df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge'], axis=1)

# Delete contributing features for 'total_minutes'
df = df.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes'], axis=1)


df.drop(['state','id'], axis=1, inplace=True)
# df.drop(['state', 'area_code', 'account_length'], axis=1, inplace=True)



###################

###ONE HOT ENCODING
df = pd.get_dummies(df, columns=['area_code'])

###################

###ONE HOT ENCODING
#df = pd.get_dummies(df, columns=['area_code','state'])


### MOVING THE Y VARIABLE TO THE END


data=np.array(df)


data[data=='no']=0
data[data=='yes']=1
data[data==False]=0
data[data==True]=1
X=data


### SPLITTING THE DATA INTO TRAIN, VALIDATION AND TEST SETS

###DATA NORMALIZATION
def normalize(X):
    X = X.astype(np.float32)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X
X=normalize(X)

# #SMOTE, oversampling the minority class (will read more about this later)
# ### Neural network

def sigmoid(x):
    return 1/(1+np.exp(-x))
def reLU(x):
    return np.maximum(0,x)


def my_dense(X,W,b, use_sigmoid=False):
   
    z=np.matmul(X,W)+b 
    
    return sigmoid(z) if use_sigmoid else reLU(z)


def forward_propagation(X,W,b):
    A=[]
    prev_A=X
    for i in range(len(W)):
        prev_A=my_dense(prev_A,W[i],b[i], use_sigmoid=(i==len(W)-1))
        A.append(prev_A)
       
    
    return A     

import pickle
weight=pickle.load(open('parameters.pkl','rb'))
W=weight[0]
b=weight[1]


y_pred=forward_propagation(X,W,b)[-1]
y_pred_new=np.where(y_pred>0.5,'yes','no')

y_pred_new = y_pred_new.flatten()

# Create an array of IDs
id_column = np.arange(1, y_pred_new.shape[0] + 1)


# Create a DataFrame
df_output = pd.DataFrame({
    'id': id_column,
    'churn': y_pred_new
})


# Save the DataFrame as a CSV file
df_output.to_csv('output.csv', index=False)