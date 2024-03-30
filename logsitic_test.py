import pickle 
import numpy as np
from logistic import features_list
with open('data_test.pkl', 'rb') as f:
    features = pickle.load(f)
