import numpy as np
import pickle
def split_features(source_data, features_list):
    X_features= np.zeros((len(source_data[0]),0))
    for i in features_list:
        X_features = np.concatenate((X_features, source_data[i]), axis=1)
    return X_features


def features_selection(features_list):
    with open('data_train.pkl', 'rb') as f:
        features = pickle.load(f)
    features_train = features[0]
    features_valid = features[2]
    X_train = split_features(features_train,features_list)
    X_validation= split_features(features_valid,features_list)
    
    y_train = features[1]
    y_validation = features[3]
    return X_train, X_validation, y_train, y_validation


def features_selection_test_set(features_list):
    with open('data_test.pkl', 'rb') as f:
        features = pickle.load(f)

    X_test = split_features(features[0],features_list)
   
    
    y_test = features[1]
    
    return X_test, y_test