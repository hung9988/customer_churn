import numpy as np
import pandas as pd
def normalize(X):
    X = X.astype(float)
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X

def load_data(df ,exclude=[], one_hot=False, normalize_=False):
    
    feature_to_exclude=exclude
    df.drop(feature_to_exclude, axis=1, inplace=True)

    #### CONVERTING yes/no to 1/0
    if one_hot==True:
        df.replace({'yes': 1, 'no': 0}, inplace=True)
        df=df.infer_objects()

        ### ONE HOT ENCODING

        df=pd.get_dummies(df, columns=[i for i in df.columns if df[i].dtype == 'object'])
        df.replace({False: 0, True: 1}, inplace=True)

        ####MOVING CHURN TO THE END

        churn = df['churn']
        df.drop('churn', axis=1, inplace=True)
        df['churn'] = churn
    data=np.array(df)
    if normalize_==True:
        data=normalize(data)
    ###COVERT TO NUMPY ARRAY
    return data
    

def data_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    np.random.shuffle(data)
    train, val, test = np.split(data, [int(train_ratio*len(data)), int((train_ratio+val_ratio)*len(data))])
    return train, val, test


def load_data_test_set(df ,exclude=[], one_hot=False, normalize_=False):
    
    feature_to_exclude=exclude
    df.drop(feature_to_exclude, axis=1, inplace=True)

    #### CONVERTING yes/no to 1/0
    if one_hot==True:
        df.replace({'yes': 1, 'no': 0}, inplace=True)
        df=df.infer_objects()

        ### ONE HOT ENCODING

        df=pd.get_dummies(df, columns=[i for i in df.columns if df[i].dtype == 'object'])
        df.replace({False: 0, True: 1}, inplace=True)

        ####MOVING CHURN TO THE END

    data=np.array(df)
    if normalize_==True:
        data=normalize(data)
    ###COVERT TO NUMPY ARRAY
    return data
    