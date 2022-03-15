import pandas as pd
import os # For reading multiple files.
import numpy as np

def load_data(path_of_data, ratio_of_training_data):

    # Listing files.
    files = os.listdir(path_of_data)
    
    # Determining the number of samples, timesteps and physical quantities.
    m = len(files)
    num_of_timesteps, num_of_features = pd.read_excel(path_of_data + '/' + files[0], header=None, skipfooter=1).values.shape
    
    # Initializing ndarrays.
    data_set_x_3D = np.zeros((m, num_of_timesteps, num_of_features))
    data_set_y = np.zeros((m, 1))
    
    # Read training data from multiple Excels.
    # Referring to https://stackoverflow.com/questions/20908018/import-multiple-excel-files-into-python-pandas-and-concatenate-them-into-one-dat
    for i in range(m):
        excel_data = pd.read_excel(path_of_data + '/' + files[i], header=None)

        # Slicing training data from excel data, converting them to ndarrays, and casting them to dtype float.
        data_set_x_3D[i, :, :] = excel_data.iloc[:-1,:].values.astype(float)

        # Slicing target data from excel data, extracting the numerical parts.
        data_set_y[i, :] = float(excel_data.iloc[-1, 0].split(':')[1])
    
    # Inserting a new axis in order to impliment Conv2D layer.
    data_set_x = data_set_x_3D[:, :, :, np.newaxis]
    
    # Splitting data set into training set and dev set.
    np.random.seed(0)
    msk = np.random.rand(m) < ratio_of_training_data
    training_set_x = data_set_x[msk]
    training_set_y = data_set_y[msk]
    dev_set_x = data_set_x[~msk]
    dev_set_y = data_set_y[~msk]
    
    return training_set_x, training_set_y, dev_set_x, dev_set_y