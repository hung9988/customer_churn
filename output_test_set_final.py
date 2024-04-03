from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df =pd.read_csv('test.csv')
# df['total_call'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']

# # Create 'total_charges' feature
# df['total_charges'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']

# # Create 'total_minutes' feature
# df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
# df = df.drop(['total_day_calls', 'total_eve_calls', 'total_night_calls'], axis=1)

# # Delete contributing features for 'total_charges'
# df = df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge'], axis=1)

# # Delete contributing features for 'total_minutes'
# df = df.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes'], axis=1)

# Drop 'churn' column

# df.drop(['state', 'area_code', 'account_length','id'], axis=1, inplace=True)

# Add 'churn' column back to the end of the DataFrame
df.drop('id',axis=1,inplace=True)
df = pd.get_dummies(df, columns=['area_code','state'])
data=np.array(df)
data[data=='no']=0
data[data=='yes']=1
data[data==False]=0
data[data==True]=1
X=data
def normalize(X):
    X = X.astype(float)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X

X = normalize(X)

weight=np.load('weights.npz',allow_pickle=True)
W,b=weight['W'],weight['b']

lr=weight['lr']
epoch=weight['epoch']
reg = weight['reg']

print(f"Learning rate: {lr}, Epochs: {epoch}, Regularization: {reg}")

### Logistic Regression



def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(X,W,b):
   
    A=np.dot(X,W)+b
    A=A.astype(float)
    return sigmoid(A)


y_pred = forward(X,W,b)
print(y_pred[0:10])
y_pred_new=np.where(y_pred>0.7,'yes','no')


id_column = np.arange(1, y_pred_new.shape[0] + 1)

# Create a DataFrame
df_output = pd.DataFrame({
    'id': id_column,
    'churn': y_pred_new
})

# Save the DataFrame as a CSV file
df_output.to_csv('output.csv', index=False)