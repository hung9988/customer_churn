import numpy as np
import pandas as pd
def normalize(X):
    X = X.astype(float)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X

def data_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    np.random.shuffle(data)
    train, val, test = np.split(data, [int(train_ratio*len(data)), int((train_ratio+val_ratio)*len(data))])
    return train, val, test

def load_data(df ,exclude=[], one_hot=False, normalize_=False,oversample=False):
    pd.set_option('future.no_silent_downcasting', True)
    feature_to_exclude=exclude
    df.drop(feature_to_exclude, axis=1, inplace=True)

    #### CONVERTING yes/no to 1/0
    if one_hot==True:
        df.replace({'yes': 1, 'no': 0}, inplace=True)
        df=df.infer_objects()

        ### ONE HOT ENCODING

        df=pd.get_dummies(df, columns=[i for i in df.columns if df[i].dtype not in ['int64', 'float64']])
        df.replace({False: 0, True: 1}, inplace=True)
 
        ####MOVING CHURN TO THE END

        churn = df['churn']
        df.drop('churn', axis=1, inplace=True)
        df['churn'] = churn
    data=np.array(df)
    if normalize_==True:
        data[:,:-1]=normalize(data[:,:-1])
        
    
    data_train,data_valid,data_test=data_split(data)
    ###COVERT TO NUMPY ARRAY
    X_train=data_train[:,:-1]
    y_train=data_train[:,-1]
    X_val=data_valid[:,:-1]
    y_val=data_valid[:,-1]
    X_test=data_test[:,:-1]
    y_test=data_test[:,-1]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    if oversample:
        weight_minority_class = np.sum(y_train == 0) / np.sum(y_train == 1)
        indices_0 = np.where(y_train == 0)[0]
        indices_1 = np.where(y_train == 1)[0]
        indices = np.concatenate([indices_0, indices_1])

        #get weights for each class
        weights = np.empty(indices_0.shape[0] + indices_1.shape[0])
        weights[:indices_0.shape[0]] = 1
        weights[indices_0.shape[0]:] = weight_minority_class
        weights = weights/np.sum(weights)

        #sample new indices
        sampled_indices = np.random.choice(indices, indices.shape[0], p=weights)

        X_train_oversampled = X_train[sampled_indices]
        y_train_oversampled = y_train[sampled_indices]
        y_train_oversampled = y_train_oversampled.reshape(-1,1)
        X_train = X_train_oversampled
        y_train = y_train_oversampled
    return X_train, y_train, X_val, y_val, X_test, y_test,df.columns
    

def total_day_eve_night_grouping(df, grouping=True):
    if grouping:
        df['total_call'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']

        # Create 'total_charges' feature
        df['total_charges'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']

        # Create 'total_minutes' feature
        df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
        df=df.drop(['total_day_calls', 'total_eve_calls', 'total_night_calls'], axis=1)

        # Delete contributing features for 'total_charges'
        df=df.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge'], axis=1)

        # Delete contributing features for 'total_minutes'
        df=df.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes'], axis=1)
    return df
    


def load_data_test_set(df ,exclude=[], one_hot=False, normalize_=False):
    pd.set_option('future.no_silent_downcasting', True)
    feature_to_exclude=exclude
    df.drop(feature_to_exclude, axis=1, inplace=True)

    #### CONVERTING yes/no to 1/0
    if one_hot==True:
        df.replace({'yes': 1, 'no': 0}, inplace=True)
        df=df.infer_objects()

        ### ONE HOT ENCODING

        df=pd.get_dummies(df, columns=[i for i in df.columns if df[i].dtype not in ['int64', 'float64']])
        df.replace({False: 0, True: 1}, inplace=True)

        ####MOVING CHURN TO THE END

    data=np.array(df)
    if normalize_==True:
        data=normalize(data)
    ###COVERT TO NUMPY ARRAY
    return data
    