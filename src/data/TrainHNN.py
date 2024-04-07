# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import itertools
import math
import random
import warnings

from HNN import HNN

# Train the model
def training_loop(X_train, y_train, model, train_indices, nn_hyps):

    # Set up
    device = nn_hyps['device']
    seed = nn_hyps['seed']
    sampling_rate = nn_hyps['sampling_rate']
    X_train, y_train = X_train.to(device), y_train.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=nn_hyps['lr'])
    criterion = torch.nn.MSELoss()

    num_epochs = nn_hyps['epochs']
    patience = nn_hyps['patience']
    tol = nn_hyps['tol']
    modelToReturn = nn_hyps['model_to_return'].lower()

    wait = 0

    if (sampling_rate == 1):
        oob_indices = train_indices
    else:
        oob_indices = [e for e in range(
            X_train.shape[0]) if e not in train_indices]

    best_epoch = 0
    best_test_loss = np.inf
    best_model = None
    train_losses = []
    test_losses = []

    # Train the model
    for epoch in range(num_epochs):
        torch.manual_seed((seed+epoch))

        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred[0][train_indices], y_train[train_indices])
        train_losses.append(loss.item())

        # Get the test loss
        model.eval()

        with torch.no_grad():
            test_output = model(X_train)
            test_loss = criterion(test_output[0][oob_indices], y_train[oob_indices])
            test_loss = test_loss.item()
            test_losses.append(test_loss)

        model.train()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            
            if modelToReturn == 'best':
                best_model = copy.deepcopy(model)
                
        else:
            wait += 1
            
        if wait > patience:
            print('Early stopped on epoch {}'.format(epoch))
            print('Best epoch: {}'.format(best_epoch))
            print('')
            
            if modelToReturn == 'last':
                best_model = copy.deepcopy(model)
                
            break

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()

    # Results
    results = {'best_model': best_model,
               'best_epoch': best_epoch,
               'train_losses': train_losses,
               'test_losses': test_losses}
    return results

# Build HNN model
def build_HNN(X_train, Y_train, nn_hyps, train_indices):

    # Build the model
    model = HNN(nn_hyps=nn_hyps)
    model = training_loop(X_train, Y_train, model, train_indices, nn_hyps)
    return model


# Bootstrap the model
def estimate_HNN(X_train, y_train, X_test, y_test, nn_hyps):

    num_bootstrap = nn_hyps['num_bootstrap']
    opt_bootstrap = nn_hyps['opt_bootstrap']
    sampling_rate = nn_hyps['sampling_rate']
    block_size = nn_hyps['block_size']
    num_hemispheres = len(nn_hyps['x_pos']) - len(nn_hyps['add_trends_to'])
    bootstrap_indices = None
    always_oob = nn_hyps['always_oob']
    seed = nn_hyps['seed']
    device = nn_hyps['device']

    # Matrix to store all predictions for every bootstrap run
    pred_in_ensemble = np.empty((X_train.shape[0], num_bootstrap))
    pred_in_ensemble[:] = np.nan
    pred_ensemble = np.empty((X_test.shape[0], num_bootstrap))
    pred_ensemble[:] = np.nan

    part_pred_in_ensemble = np.empty((X_train.shape[0], num_hemispheres, num_bootstrap))
    part_pred_in_ensemble[:] = np.nan
    part_pred_ensemble = np.empty((X_test.shape[0], num_hemispheres, num_bootstrap))
    part_pred_ensemble[:] = np.nan

    gaps_in_ensemble = np.empty((X_train.shape[0], num_hemispheres, num_bootstrap))
    gaps_in_ensemble[:] = np.nan
    gaps_ensemble = np.empty((X_test.shape[0], num_hemispheres, num_bootstrap))
    gaps_ensemble[:] = np.nan

    trends_in_ensemble = np.empty( (X_train.shape[0], len(nn_hyps['add_trends_to']), num_bootstrap))
    trends_in_ensemble[:] = np.nan
    trends_ensemble = np.empty( (X_test.shape[0], len(nn_hyps['add_trends_to']), num_bootstrap))
    trends_ensemble[:] = np.nan

    oob_pos = np.empty((X_train.shape[0], num_bootstrap))
    oob_pos[:] = np.nan

    boot_pos = np.empty((X_train.shape[0], num_bootstrap))
    boot_pos[:] = np.nan

    losses = [None]*num_bootstrap

    best_epochs = np.empty(num_bootstrap)
    best_epochs[:] = np.nan

    # Bootstrap the model
    for i in range(num_bootstrap):

        # set seed
        np.random.seed(seed+i)

        print('Bootstrap {}'.format(i))

        if opt_bootstrap == 1:  # Individual obs bootstrap

            k = int(sampling_rate * X_train.shape[0])

            boot = sorted(random.sample(
                list(range(X_train.shape[0])), k=k))

            # remove always_oob from the bootrap sample
            if always_oob is not None:
                boot = [e for e in boot if e not in always_oob]

            oob = [e for e in list(
                range(X_train.shape[0])) if e not in boot]
            oos = list(
                range(X_train.shape[0], X_train.shape[0] + X_test.shape[0]))

        if opt_bootstrap == 2:  # Block bootstrap

            n_obs = X_train.shape[0]
            
            # Select the size of first block
            first_block_size = random.sample(list(range(int(block_size / 2), block_size + 1)), k=1)[0]
            
            # Get the starting ids of the blocks
            block_start_ids = [0] + list(range(first_block_size, n_obs, block_size))

            # If last block size < half of block size
            last_block_size = n_obs - block_start_ids[-1]
            if last_block_size < block_size / 2:
                block_start_ids.remove(block_start_ids[-1])

            num_oob_blocks = int(((1-sampling_rate) * n_obs) / block_size)
            oob_blocks = random.sample(list(range(len(block_start_ids))), k=num_oob_blocks)

            # Get the OOB indices
            oob = list(itertools.chain(*[list(range(block_start_ids[e], block_start_ids[e+1])) if e < len(block_start_ids) - 1 else list(range(block_start_ids[e], n_obs))
                                         for e in oob_blocks]))
            boot = [e for e in list(range(n_obs)) if e not in oob]

            # remove always_oob from the bootrap sample
            if always_oob is not None:
                boot = [e for e in boot if e not in always_oob]
                oob = oob + always_oob
                
                # keep oob unique values
                oob = list(set(oob))

            oos = list(range(X_train.shape[0], X_train.shape[0] + X_test.shape[0]))

        if sampling_rate == 1:
            boot = sorted(random.sample(list(range(X_train.shape[0])), k=k))
            oob = range(X_train.shape[0])
            oos = list(range(X_train.shape[0], X_train.shape[0] + X_test.shape[0]))

        # Train
        results = build_HNN(X_train=X_train, Y_train=y_train,
                            nn_hyps=nn_hyps, train_indices=boot)
        boot_model = copy.deepcopy(results['best_model']).to(device)

        # Get the predictions
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        boot_model.eval()
        with torch.no_grad():

            # HNN predictions
            pred_in_ensemble[oob, i] = boot_model(X_train[oob, ])[0].detach().cpu().numpy()
            pred_ensemble[:, i] = boot_model(X_test)[0].detach().cpu().numpy()
            part_pred_in_ensemble[oob, :, i] = boot_model(X_train[oob, ])[1].detach().cpu().numpy()
            part_pred_ensemble[:, :, i] = boot_model(X_test)[1].detach().cpu().numpy()
            gaps_in_ensemble[oob, :, i] = boot_model(X_train[oob, ])[3].detach().cpu().numpy()
            gaps_ensemble[:, :, i] = boot_model(X_test)[3].detach().cpu().numpy()
            trends_in_ensemble[oob, :, i] = boot_model(X_train[oob, ])[2].detach().cpu().numpy()
            trends_ensemble[:, :, i] = boot_model(X_test)[2].detach().cpu().numpy()

            # Bootstrap parameters and metrics
            oob_pos[oob, i] = oob
            boot_pos[boot, i] = boot
            best_epochs[i] = results['best_epoch']
            losses[i] = pd.DataFrame( {'Train': results['train_losses'], 'Validation': results['test_losses']})

        # Get bootstraps mean
        pred_in_mean = np.nanmean(pred_in_ensemble, axis=1)
        pred_mean = np.nanmean(pred_ensemble, axis=1)

        # Get the mean of part_pred_in_ensemble and part_pred_ensemble for each part
        part_pred_in_mean = np.empty((part_pred_in_ensemble.shape[0], part_pred_in_ensemble.shape[1]))
        part_pred_in_mean[:] = np.nan
        part_pred_mean = np.empty((part_pred_ensemble.shape[0], part_pred_ensemble.shape[1]))
        part_pred_mean[:] = np.nan

        for i in range(part_pred_in_mean.shape[1]):
            part_pred_in_mean[:, i] = np.nanmean(part_pred_in_ensemble[:, i, :], axis=1)
            
            part_pred_mean[:, i] = np.nanmean(part_pred_ensemble[:, i, :], axis=1)

        # Get the mean of gaps_in_ensemble and gaps_ensemble for each part
        gaps_in_mean = np.empty((gaps_in_ensemble.shape[0], gaps_in_ensemble.shape[1]))
        gaps_in_mean[:] = np.nan
        gaps_mean = np.empty((gaps_ensemble.shape[0], gaps_ensemble.shape[1]))
        gaps_mean[:] = np.nan

        for i in range(gaps_in_mean.shape[1]):
            gaps_in_mean[:, i] = np.nanmean(gaps_in_ensemble[:, i, :], axis=1)
            gaps_mean[:, i] = np.nanmean(gaps_ensemble[:, i, :], axis=1)

        # Get the mean of trends_in_ensemble and trends_ensemble for each part
        trends_in_mean = np.empty( (trends_in_ensemble.shape[0], trends_in_ensemble.shape[1]))
        trends_in_mean[:] = np.nan
        trends_mean = np.empty((trends_ensemble.shape[0], trends_ensemble.shape[1]))
        trends_mean[:] = np.nan

        for i in range(trends_in_mean.shape[1]):
            trends_in_mean[:, i] = np.nanmean(trends_in_ensemble[:, i, :], axis=1)
            trends_mean[:, i] = np.nanmean(trends_ensemble[:, i, :], axis=1)

    # Final output
    return {'pred_in_ensemble': pred_in_ensemble,
            'pred_ensemble': pred_ensemble,
            'pred_in_mean': pred_in_mean,
            'pred_mean': pred_mean,
            'part_pred_in_ensemble': part_pred_in_ensemble,
            'part_pred_ensemble': part_pred_ensemble,
            'part_pred_in_mean': part_pred_in_mean,
            'part_pred_mean': part_pred_mean,
            'gaps_in_ensemble': gaps_in_ensemble,
            'gaps_ensemble': gaps_ensemble,
            'gaps_in_mean': gaps_in_mean,
            'gaps_mean': gaps_mean,
            'trends_in_ensemble': trends_in_ensemble,
            'trends_ensemble': trends_ensemble,
            'trends_in_mean': trends_in_mean,
            'trends_mean': trends_mean,
            'oob_pos': oob_pos,
            'boot_pos': boot_pos,
            'best_epochs': best_epochs,
            'losses': losses}