import numpy as np

def feature_selection(features_list, source_data):
    X_featues= np.zeros((len(source_data[0]),0))
    for i in features_list:
        X_featues = np.concatenate((X_featues, source_data[i]), axis=1)
    return X_featues