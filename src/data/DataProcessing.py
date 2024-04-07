from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

# Split data into train and test
def preprocess_data(data, split_id = 149, scale_y = True, scale_x = True):
  X = data.iloc[:, 1:].to_numpy()
  y = data['y'].to_numpy()

  # Train test split (70-30, same as the ML code)
  X_train = X[:split_id, :]
  y_train = y[:split_id]
  X_test = X[split_id:, :]
  y_test = y[split_id:]

  y_mean = y_train.mean()
  y_std = y_train.std()

  # Standardize the variables
  if scale_x == True:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

  if scale_y == True:
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

  # Convert to tensor
  X_train = torch.tensor(X_train, dtype = torch.float)
  X_test = torch.tensor(X_test, dtype = torch.float)
  y_train = torch.tensor(y_train, dtype = torch.float)
  y_test = torch.tensor(y_test, dtype = torch.float)

  return X_train, X_test, y_train, y_test, y_mean, y_std

  # create train and test loader
def create_data_loader(X_train, y_train, X_test, y_test, batch_size = 32):
    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)
    train_loader = Data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = Data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    return train_loader, test_loader
  
# Function that inverse the scaling
def inverse_scaling(y, y_mean, y_std):
    y = y*y_std + y_mean
    return y

# Function that return position of columns
def find_position(col_names, string_col):
    pos = []
    for i in string_col:
        posCol = col_names.index(i)
        pos.append(posCol)
    return pos

# Function that append two list of string
def append_list(list1, list2):
    list3 = []
    for i in list1:
        for j in list2:
            list3.append(i+j)
    return list3

# MSE function
def MSE(y_true, y_pred):
    return np.nanmean((y_true - y_pred)**2)