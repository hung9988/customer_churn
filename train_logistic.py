import numpy as np
import pandas as pd
from data_process import data_process
from fit_logistic import fit_logistic,accuracy,loss,forward
df=pd.read_csv('train.csv')
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

churn = df['churn']

# Drop 'churn' column
df = df.drop('churn', axis=1)

# Add 'churn' column back to the end of the DataFrame
df['churn'] = churn
features_columns = df.columns[:-1]

data=np.array(df)
features_list= [col for col in features_columns if col not in ['state', 'area_code', 'account_length']]
X_train, y_train, X_valid, y_valid, X_test, y_test = data_process(data,0.7,0.15,0.15,features_columns,features_list)




##TRAINING PART###
lr_value=[0.01,0.05,0.1,0.5,1,2,10]
reg_value=[0.01,0.05,0.1,0.5,1,2,10]
epochs=[1000,5000,10000,20000]
max_accuracy=0
default_epoch=10000
default_lr=0.1
default_reg=0.1
with open('output_engineered_features.txt', 'w') as f:
    pass
for lr in lr_value:

   W, b, costs = fit_logistic(X_train, y_train, lr, default_epoch, default_reg)
   y_hat = forward(X_valid, W, b)
   if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
   with open('output_engineered_features.txt', 'a') as f:
        
        
        f.write(f"Training with epoch={default_epoch}, lr={lr}, reg={default_reg}\n")
       
        f.write("-------------------------------------------------\n")
        
        

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

   W, b, costs = fit_logistic(X_train, y_train, default_lr, epoch, default_reg)
   y_hat = forward(X_valid, W, b)
   if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
   with open('output_engineered_features.txt', 'a') as f:
       
        
        f.write(f"Training with epoch={epoch}, lr={default_lr}, reg={default_reg}\n")
       
        f.write("-------------------------------------------------\n")
        
        

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

   W, b, costs = fit_logistic(X_train, y_train, default_lr, default_epoch, reg)
   y_hat = forward(X_valid, W, b)
   if accuracy(W, b, X_valid, y_valid)>max_accuracy:
            W_best_fit=W
            b_best_fit=b
            costs_best_fit=costs
   with open('output_engineered_features.txt', 'a') as f:
       
        
        f.write(f"Training with epoch={default_epoch}, lr={default_lr}, reg={reg}\n")
       
        f.write("-------------------------------------------------\n")
        
        

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
np.savez('weights_engineered_features.npz', W=W_best_fit, b=b_best_fit,costs=costs_best_fit)


