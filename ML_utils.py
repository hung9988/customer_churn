import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(df ,exclude=[], one_hot=False, normalize_=False,oversample=False):
    pd.set_option('future.no_silent_downcasting', True)
    feature_to_exclude=exclude
    df.drop(feature_to_exclude, axis=1, inplace=True)

    if normalize_==True:
        scaler=StandardScaler()
        for i in df.columns:
            if df[i].dtype in ['int64', 'float64']:
                df[i] = scaler.fit_transform(df[[i]])
                
     #### CONVERTING yes/no to 1/0    
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    df=df.infer_objects()
    #### CONVERTING yes/no to 1/0
    if one_hot==True:
        
        ### ONE HOT ENCODING

        df=pd.get_dummies(df, columns=[i for i in df.columns if df[i].dtype not in ['int64', 'float64']])
        df.replace({False: 0, True: 1}, inplace=True)
 
        ####MOVING CHURN TO THE END
    churn = df['churn']
    df.drop('churn', axis=1, inplace=True)
    df['churn'] = churn
        
    data=np.array(df)
    
    X=data[:, :-1]
    y=data[:, -1]
    y=y.astype('int')
    

    if oversample==True:

        smote = SMOTE(sampling_strategy='minority')
        X,y = smote.fit_resample(X, y)
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    
    return X_train, y_train.reshape(-1,1), X_test, y_test.reshape(-1,1), df.columns
    

def total_charges_grouping(df, grouping=True):
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
