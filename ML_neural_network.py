import numpy as np
def reLU(z):
    return np.maximum(0,z)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def reLU_derivative(z):
    return z>0


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


#### -0.5 is for preventing the weights from being too large, causing overflow of sigmoid
def initialize_parameters(model):
    W=[]
    b=[]
    for i in range(1,len(model)):
        W.append(np.random.rand(model[i-1],model[i]) - 0.5)
        b.append(np.random.rand(1,model[i]) - 0.5)
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

def back_propagation(X,y,A,W,alpha=0.01):
    dW=[]
    db=[]
    m=y.shape[0]
    
    #### CALCULATING dC/dA
    dA=(A[-1]-y)
    #### CALCULATING dC/dZ by using the chain rules
    dZ=dA*sigmoid_derivative(A[-1])
    
    #### CALCULATING dC/dW and alpha * W is the regularization term
    dC_dW = (A[-2].T.dot(dZ)+alpha * W[-1])/m
    dW.append(dC_dW)
    
    #### CALCULATING dC/db
    
    dC_db=np.sum(dZ,axis=0,keepdims=True)/m
    db.append(dC_db)
  
  
    for i in range(len(W)-2,0,-1):
       
        dC_dA=dZ.dot(W[i+1].T)
        dA_dZ=reLU_derivative(A[i])
        dZ=dC_dA*dA_dZ
        dW.append((A[i-1].T.dot(dZ)+alpha * W[i])/m)
        db.append(np.sum(dZ,axis=0,keepdims=True)/m)
        
    ### split to dot product with X
    dZ=dZ.dot(W[1].T)*reLU_derivative(A[0])
    dW.append((X.T.dot(dZ)+alpha * W[0])/m)
    db.append(np.sum(dZ,axis=0,keepdims=True)/m)
    
    ### return the inverse of dW and db because we calculated them in reverse order
    return dW[::-1],db[::-1]


def train(X_train,y_train,X_valid,y_valid,learning_rate,epochs,alpha,model):
    costs_train=[]
    costs_valid=[]
    W,b=initialize_parameters(model)
    for i in range(epochs):
        A=forward_propagation(X_train,W,b)
        dW,db=back_propagation(X_train,y_train,A,W,alpha)
        W,b=update_parameters(W,b,dW,db,learning_rate)
        loss=squared_error(y_train,A[-1])
        loss_valid=squared_error(y_valid,forward_propagation(X_valid,W,b)[-1])
        costs_train.append(loss)
        
        costs_valid.append(loss_valid)
      
        if i%1000==0:
            acc=accuracy(y_train,A[-1])
            print(f"Epoch {i} : Loss {loss}")
            print(f"Train Accuracy: {acc}")
            acc_valid=accuracy(y_valid,forward_propagation(X_valid,W,b)[-1])
            print(f"Validation Loss: {loss_valid}")
            print(f"Validation Accuracy: {acc_valid}")
        
        if acc == 1:
            break
        
        
        
    return W,b,costs_train,costs_valid,acc_valid

def accuracy(y,A,threshold=0.5):
    return np.mean((A>threshold)==y)


                        
def grid_search(X_train, y_train, X_val, y_val, model_list, lr_list, alpha_list, threshold_list, epoch_list):
        best_accuracy = 0
        best_model=[]
        layer_cost_train = [] 
        layer_cost_valid=[] 
        model_values = [] 
       

        for model in model_list:
            for lr in lr_list:
                for alpha in alpha_list:
                    for epoch in epoch_list:
                        for threshold in threshold_list:
                            W,b,costs_train,costs_valid,acc_valid=train(X_train,y_train,X_val,y_val,lr,epoch,alpha,model)
                            model_values.append({"model": model, "lr": lr, "alpha": alpha, "epoch": epoch, "threshold": threshold})
                            layer_cost_train.append([costs_train])
                            layer_cost_valid.append([costs_valid])
                            
                
                            if acc_valid > best_accuracy:
                                best_accuracy = acc_valid
                                best_lr = lr
                                best_alpha = alpha
                                best_threshold = threshold
                                W_best = W
                                b_best = b
                                best_model=model
                                best_epoch=epoch
        
        print(f"Best Accuracy: {best_accuracy}")
        print(f"Best Model: {best_model}") 
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Learning Rate: {best_lr}")
        print(f"Best Alpha: {best_alpha}")
        print(f"Best Threshold: {best_threshold}")
        
        
        return {"best_accuracy": best_accuracy, "best_model": best_model,"best_epoch":best_epoch, "best_lr": best_lr, "best_alpha": best_alpha, "best_threshold": best_threshold, "W_best": W_best, "b_best": b_best, "layer_cost_train": layer_cost_train, "layer_cost_valid": layer_cost_valid, "model_values": model_values}
                                