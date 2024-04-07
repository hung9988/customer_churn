import numpy as np
### LOGISTIC REGRESSION IMPLEMENTATION

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X,W,b):
    A=np.dot(X,W)+b

    return sigmoid(A)


### CROSS ENTROPY LOSS FUNCTION, USE TO EVALUATE THE MODEL
def loss(y,y_hat):
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))


### GRADIENT DESCENT IMPLEMENTATION, USE TO TRAIN THE MODEL

def gradient_descent(X,y,X_valid,y_valid,W,b,learning_rate,epochs,regularization_term):
    costs_train=[]
    costs_valid=[]
    for epoch in range(epochs):
        
        y_hat = forward(X,W,b)
        y_hat=y_hat.reshape(-1,1)
        l=loss(y,y_hat)
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: {l}')
            
        costs_train.append(l)
        y_hat_valid=forward(X_valid,W,b)
        y_hat_valid=y_hat_valid.reshape(-1,1)
        l_valid=loss(y_valid,y_hat_valid)
        costs_valid.append(l_valid)
    
        Z=y_hat-y
        W=W.reshape(-1,1)

        dW = (np.dot(X.T, Z) + regularization_term * W)/X.shape[0]
      
        db=np.mean(Z)
       
        W=W-learning_rate*dW
        b=b-learning_rate*db
        
    return W,b,costs_train,costs_valid



def fit_logistic(X, y,X_valid,y_valid, lr, epochs, reg):
    W,b=np.random.rand(X.shape[1]),np.random.rand(1)
    W_fit,b_fit,costs_train,costs_valid = gradient_descent(X, y,X_valid,y_valid, W, b, lr, epochs, reg)
    return W_fit, b_fit, costs_train,costs_valid


### PREDICTION AND ACCURACY FUNCTIONS

def predict(X,W,b,threshold=0.5):
    y_hat=forward(X,W,b)
    y_hat[y_hat>=threshold]=1
    y_hat[y_hat<threshold]=0
    return y_hat

def accuracy(y,y_hat):
    return np.sum(y==y_hat)/y.shape[0]
    
    

def grid_search(X_train,y_train,X_val,y_val,lr_value,reg_value,epoch_value,threshold_value):
   max_accuracy=0
   best_lr=0
   best_reg=0
   best_epoch=0
   layer_cost_train = [] 
   layer_cost_valid=[] 
   model_parameters=[]
   for epoch in epoch_value:
      for lr in lr_value:
         for reg in reg_value:
            for threshold in threshold_value:
      ###FITTING THE MODEL
               
               W, b, costs_train,costs_valid = fit_logistic(X_train, y_train,X_val,y_val, lr, epoch, reg)
               layer_cost_train.append([costs_train])
               layer_cost_valid.append([costs_valid])
               model_parameters.append({'W':W,'b':b,'lr':lr,'reg':reg,'epoch':epoch,'threshold':threshold})
                            
      ### USING W and b to predict the validation set
               y_hat = predict(X_val, W, b, threshold=threshold)
               acc_val=accuracy(y_val,y_hat)
      ### CALCULATING THE ACCURACY, if the accuracy improved, we save the variables
               if acc_val>max_accuracy:
                  W_best_fit=W
                  b_best_fit=b
                  
                  max_accuracy=acc_val
                  best_lr=lr
                  best_reg=reg
                  best_epoch=epoch
                  best_threshold=threshold
                  
   return {'W':W_best_fit,'b':b_best_fit,'lr':best_lr,'reg':best_reg,'epoch':best_epoch,'threshold':best_threshold,'accuracy':max_accuracy,'cost_train':layer_cost_train,'cost_valid':layer_cost_valid,'model_parameters':model_parameters}
