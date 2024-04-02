import numpy as np
def init_weight_and_bias(n):
    return np.random.rand(n), np.random.rand(1)

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

def accuracy(W,b,X,y,threshold=0.5):
    y_h = forward(X, W, b)
    y_h[y_h >= threshold] = 1
    y_h[y_h < threshold] = 0
    correct = np.sum(y_h == y)
    return correct/len(y)

def fit_logistic(X, y, lr, epochs, reg):
    W_fit, b_fit = init_weight_and_bias(X.shape[1])
    
    W_fit, b_fit, costs = train(X, y, W_fit, b_fit, lr, epochs, reg)
    return W_fit, b_fit, costs

