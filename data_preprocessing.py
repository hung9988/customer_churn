import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
raw_data = pd.read_csv('train.csv')
pretty_df = raw_data.head(10).to_string(index=False)

# Print the pretty string
print(pretty_df)

# Continue with your code
raw_data = np.array(raw_data)
#np data
# raw_data= np.array(raw_data)

# split = 0.8

# train_set = raw_data[:int(split*len(raw_data))]
# validation_set = raw_data[int(split*len(raw_data)):]

# X_train = train_set[:,:-1]
# y_train = train_set[:,-1]
# X_validation = validation_set[:,:-1]
# y_validation = validation_set[:,-1]


