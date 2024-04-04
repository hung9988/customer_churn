from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


df =pd.read_csv('train.csv')

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


df.drop(['state'], axis=1, inplace=True)
# df.drop(['state', 'area_code', 'account_length'], axis=1, inplace=True)



###################

###ONE HOT ENCODING


df = pd.get_dummies(df, columns=['area_code'])


### MOVING THE Y VARIABLE TO THE END
churn = df['churn']
df = df.drop('churn', axis=1)
df['churn'] = churn


data=np.array(df)


data[data=='no']=0
data[data=='yes']=1
data[data==False]=0
data[data==True]=1

X=data[:,:-1]
y=data[:,-1]



### SPLITTING THE DATA INTO TRAIN, VALIDATION AND TEST SETS

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 

###DATA NORMALIZATION
def normalize(X):
    X = X.astype(float)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


#SMOTE, oversampling the minority class (will read more about this later)
# X_train_oversampled_smote = []
# labels_train_oversampled_smote = []
# indices_0 = np.where(y_train == 0)[0]
# indices_1 = np.where(y_train == 1)[0]
# indices = np.concatenate([indices_0, indices_1])
# for _ in range(X_train.shape[0]):
#     p = np.random.random()
#     #sample from majority class
#     if p < 0.5:
#         X_train_oversampled_smote.append(X_train[np.random.choice(indices_0)])
#         labels_train_oversampled_smote.append(0)
#     #sample from minority class
#     else:
#         #get two random samples from minority class
#         minority_samp_1 = X_train[np.random.choice(indices_1)]
#         minority_samp_2 = X_train[np.random.choice(indices_1)]
        
#         #get random proportion with which to mix them
#         prop = np.random.random()
        
#         #generate synthetic sample from minority class
#         synthetic_minority_samp = prop*minority_samp_1 + (1-prop)*minority_samp_2
#         X_train_oversampled_smote.append(synthetic_minority_samp)
#         labels_train_oversampled_smote.append(1)
        
# X_train = np.array(X_train_oversampled_smote)
# y_train = np.array(labels_train_oversampled_smote)

# print(X_train[0])
# print(X_train.shape)




###CONVERT TO APPROPIATE FORMAT
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

y_train = y_train.reshape(-1,1)
####################

# Assuming y_train and y_val are numpy arrays or pandas series
# plt.figure(figsize=(10, 6))

# plt.subplot(1, 2, 1)
# plt.hist(y_train_pre, bins=[-0.5, 0.5, 1.5], edgecolor='black')
# plt.title('y_train_before_SMOTE')
# plt.xticks([0, 1])
# plt.xlabel('Class')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# plt.hist(y_train, bins=[-0.5, 0.5, 1.5], edgecolor='black')
# plt.title('y_train_after_SMOTE')
# plt.xticks([0, 1])
# plt.xlabel('Class')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()


####



def reLU(z):
    return np.maximum(0,z)
def reLU_derivative(z):
    return z>0
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


def initialize_parameters(layers):
    W=[]
    b=[]
    for i in range(1,len(layers)):
        W.append(np.random.rand(layers[i-1],layers[i]) - 0.5)
        b.append(np.random.rand(1,layers[i]) - 0.5)
        
        
    return W,b



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
    
def squared_error(y,A):
    return np.sum((y-A)**2)/y.shape[0]

def update_parameters(W,b,dW,db,learning_rate):
    for i in range(len(W)):
        W[i]=W[i]-learning_rate*dW[i]
        b[i]=b[i]-learning_rate*db[i]
        
    return W,b

def train(X_train,y_train,learning_rate,epochs,layers):
    W,b=initialize_parameters(layers)
    for i in range(epochs):
        A=forward_propagation(X_train,W,b)
        dW,db=back_propagation(X_train,y_train,A,W)
        W,b=update_parameters(W,b,dW,db,learning_rate)
        # print(f"Epoch {i} : Loss {squared_error(y_train,A[-1])}")
        # print(f"Train Accuracy: {accuracy(y_train,A[-1])}")
    return W,b

def back_propagation(X,y,A,W,alpha=0.01):
    dW=[]
    db=[]
    m=y.shape[0]
    dA=(A[-1]-y)

    dZ=dA*sigmoid_derivative(A[-1])
    dC_dW = (A[-2].T.dot(dZ)+alpha * W[-1])/m

    
    dW.append(dC_dW)
    dC_db=np.sum(dZ,axis=0,keepdims=True)/m
    db.append(dC_db)
  
    for i in range(len(W)-2,0,-1):
       
        U=dZ.dot(W[i+1].T)
        V=reLU_derivative(A[i])
        dZ=U*V
        dW.append((A[i-1].T.dot(dZ)+alpha * W[i])/m)
        db.append(np.sum(dZ,axis=0,keepdims=True)/m)
        
    dZ=dZ.dot(W[1].T)*reLU_derivative(A[0])
    dW.append((X.T.dot(dZ)+alpha * W[0])/m)
    db.append(np.sum(dZ,axis=0,keepdims=True)/m)
    return dW[::-1],db[::-1]


def accuracy(y,A,threshold=0.5):
    return np.mean((A>threshold)==y)



layer_list=[[X_train.shape[1],64,64,1],[X_train.shape[1],64,128,128,64,1],[X_train.shape[1],64,64,64,1]]
lr_list = [0.005,0.01,0.05,0.1,0.5,0.7]
reg_list = [0.005,0.01,0.05,0.1,0.5,0.7]
threshold_list = [0.4,0.5,0.6,0.7,0.8,0.9]
default_epoch = 10000
best_accuracy = 0
best_layer = []

for layers in layer_list:
    for lr in lr_list:
        for reg in reg_list:
            for threshold in threshold_list:
                W,b=train(X_train,y_train,lr,default_epoch,layers)
                A=forward_propagation(X_val,W,b)
                acc = accuracy(y_val,A[-1],threshold)
                
                
                # acc_test = accuracy(y_test,forward_propagation(X_test,W,b)[-1],threshold)
                # print(f"Test Accuracy: {acc_test}")
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_lr = lr
                    best_reg = reg
                    best_threshold = threshold
                    W_best = W
                    b_best = b
                    best_layer = layers
                    
                
            
                
print(f"Best Accuracy on validation set: {best_accuracy}")    
print(f"Best Learning Rate: {best_lr}")
print(f"Best Regularization: {best_reg}")
print(f"Best Threshold: {best_threshold}")
print(f"Best Layer: {best_layer}")

A_test=forward_propagation(X_test,W_best,b_best)
acc_test = accuracy(y_test,A_test[-1],best_threshold)
print(f"Test Accuracy with W,b: {acc_test}")


import pickle

# Save W and b to a single file
with open('parameters.pkl', 'wb') as f:
    pickle.dump((W_best,b_best ), f)