import pandas as pd
import os # For reading multiple files.
import numpy as np

def load_data(path_to_train_and_dev, ratio_of_train_data, path_to_test):

    # Listing files.
    files_train_and_dev = os.listdir(path_to_train_and_dev)
    files_test = os.listdir(path_to_test)
    
    # Determining the number of samples, timesteps and physical quantities.
    m_train_and_dev = len(files_train_and_dev)
    m_test = len(files_test)
    num_of_timesteps, num_of_features = pd.read_excel(path_to_train_and_dev + '/' + files_train_and_dev[0], header=None, skipfooter=1).values.shape
    
    # Initializing ndarrays.
    train_and_dev_set_x = np.zeros((m_train_and_dev, num_of_timesteps, num_of_features))
    train_and_dev_set_y = np.zeros((m_train_and_dev, 1))
    test_set_x = np.zeros((m_test, num_of_timesteps, num_of_features))
    
    # Referring to https://stackoverflow.com/questions/20908018/import-multiple-excel-files-into-python-pandas-and-concatenate-them-into-one-dat
    # Read training and dev data from multiple Excels.
    for i in range(m_train_and_dev):
        excel_data = pd.read_excel(path_to_train_and_dev + '/' + files_train_and_dev[i], header=None)
        
        # Slicing training data from excel data, converting them to ndarrays, and casting them to dtype float.
        train_and_dev_set_x[i, :, :] = excel_data.iloc[:-1,:].values.astype(float)
        
        # Slicing target data from excel data, extracting the numerical parts.
        train_and_dev_set_y[i, :] = float(excel_data.iloc[-1, 0].split(':')[1])
        
    # Read test data from multiple Excels.
    for i in range(m_test):
        excel_data = pd.read_excel(path_to_test + '/' + files_test[i], header=None)
        
        # Converting test data to ndarrays, and casting them to dtype float.
        test_set_x[i, :, :] = excel_data.values.astype(float)
    
    # Splitting data set into training set and dev set.
    np.random.seed(0)
    msk = np.random.rand(m_train_and_dev) < ratio_of_train_data
    train_set_x = train_and_dev_set_x[msk]
    train_set_y = train_and_dev_set_y[msk]
    dev_set_x = train_and_dev_set_x[~msk]
    dev_set_y = train_and_dev_set_y[~msk]
    
    # Inserting new axes in order to impliment Conv2D layer.
    train_set_x = train_set_x[:, :, :, np.newaxis]
    dev_set_x = dev_set_x[:, :, :, np.newaxis]
    test_set_x = test_set_x[:, :, :, np.newaxis]
    
    return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x