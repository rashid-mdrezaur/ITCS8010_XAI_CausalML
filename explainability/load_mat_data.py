#!/usr/bin/env python3

import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_mat = scipy.io.loadmat('data/model3_inputs.mat')['model3_inputs']
target_mat = scipy.io.loadmat('data/bldgs_targets.mat')['bldgs_targets']

# TODO: Figure out column meanings. Check if any are continuous or all are discrete
#  Matlab script uses MinMax for all columns
scaler = MinMaxScaler()
input_mat = scaler.fit_transform(input_mat)

data = pd.DataFrame(input_mat)
data['target'] = np.argmax(target_mat, axis=1)

data.to_csv('data/data_df.csv', index=False, header=True)
