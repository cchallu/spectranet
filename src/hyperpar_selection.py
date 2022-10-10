# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as t
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.ts_dataset import TsDataset
from .models.spectranet import SpectraNet


def mae(y, y_hat, weights):

    mae = np.average(np.abs(y - y_hat), weights=weights)
    
    return mae

def get_datasets(data,
                 ts_val,
                 ts_test,
                 mask):

    # Train data
    train_data = data.copy()[['unique_id', 'ds', 'y']]
    train_data.reset_index(drop=True, inplace=True)

    train_data['available_mask'] = 1
    train_data['sample_mask'] = 1

    idx_out = train_data.groupby('unique_id').tail(ts_val+ts_test).index
    train_data.loc[idx_out, 'sample_mask'] = 0

    # Test data
    test_data = train_data.copy()
    test_data['sample_mask'] = 0
    idx_test = test_data.groupby('unique_id').tail(ts_test).index
    test_data.loc[idx_test, 'sample_mask'] = 1

    # Validation data
    val_data = train_data.copy()
    val_data['sample_mask'] = 1
    val_data['sample_mask'] = val_data['sample_mask'] - train_data['sample_mask']
    val_data['sample_mask'] = val_data['sample_mask'] - test_data['sample_mask']

    # Available mask
    if mask is not None:
        train_data['available_mask'] = train_data['available_mask']*mask.flatten()
        val_data['available_mask'] = val_data['available_mask']*mask.flatten()
        test_data['available_mask'] = test_data['available_mask']*mask.flatten()

    return train_data, val_data, test_data

def instantiate_model(mc):

    lr_decay_step_size = int(mc['max_steps'] / mc['n_lr_decays'])
    z_t_dim = int((mc['n_time_in']+mc['n_time_out'])*mc['z_t_dim_frac'])

    model =  SpectraNet(n_time_in=int(mc['n_time_in']),
                        n_time_out=int(mc['n_time_out']),
                        n_features=int(mc['n_features']),
                        n_layers=int(mc['n_layers']),
                        n_filters_multiplier=int(mc['n_filters_multiplier']),
                        max_filters=int(mc['max_filters']),
                        kernel_size=int(mc['kernel_size']),
                        stride=int(mc['stride']),
                        dilation=int(mc['dilation']),
                        z_t_dim=z_t_dim,
                        n_polynomial=int(mc['n_polynomial']),
                        n_harmonics=int(mc['n_harmonics']),
                        z_iters=int(mc['z_iters']),
                        z_sigma=mc['z_sigma'],
                        z_step_size=mc['z_step_size'],
                        z_with_noise=mc['z_with_noise'],
                        z_persistent=mc['z_persistent'],
                        normalize_windows=mc['normalize_windows'],
                        forecasting_mask=mc['forecasting_mask'],
                        noise_std=mc['noise_std'],
                        learning_rate=mc['learning_rate'],
                        lr_decay=mc['lr_decay'],
                        lr_decay_step_size=lr_decay_step_size,
                        weight_decay=mc['weight_decay'],
                        random_seed=int(mc['random_seed']))

    return model

def predict(mc,
            model,
            loader):

    y_true = []
    y_hat = []
    mask = []
    model.model.z_iters = mc['z_iters_inference']
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model.to(device)

    for batch in tqdm(loader):
        y_hat_batch = []
        for i in range(mc['inference_repeats']):
            y_true_batch, y_hat_batch_run, mask_batch, *_ = model(batch)
            y_true_batch = y_true_batch.detach().cpu()
            y_hat_batch_run = y_hat_batch_run.detach().cpu()
            mask_batch = mask_batch.detach().cpu()
            y_hat_batch.append(y_hat_batch_run)
        
        y_hat_batch = t.stack(y_hat_batch)
        y_hat_batch = t.median(y_hat_batch, dim=0)[0]

        y_true.append(y_true_batch)
        y_hat.append(y_hat_batch)
        mask.append(mask_batch)
    
    y_true = t.cat(y_true).numpy() 
    y_hat = t.cat(y_hat).numpy() 
    mask = t.cat(mask).numpy() 

    return y_true, y_hat, mask

def run_model(mc,
              data,
              ts_val,
              ts_test,
              mask,
              trials,
              results_file):

    if (len(trials) % 5):
        with open(results_file, "wb") as f:
            pickle.dump(trials, f)

    assert ts_test % mc['val_sample_freq']==0, 'outsample size should be multiple of val_sample_freq'

    data = data.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    start = time.time()

    # ----------------------------------------- Dataset and Loaders ---------------------------------- #
    train_data, val_data, test_data = get_datasets(data=data, ts_val=ts_val, ts_test=ts_test, mask=mask)

    train_data = train_data.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    val_data = val_data.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    test_data = test_data.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    train_dataset = TsDataset(data=train_data,
                              input_size=int(mc['n_time_in']),
                              output_size=int(mc['n_time_out']),
                              sample_freq=int(mc['sample_freq']))
    val_dataset   = TsDataset(data=val_data,
                              input_size=int(mc['n_time_in']),
                              output_size=int(mc['n_time_out']),
                              sample_freq=int(mc['val_sample_freq']))
    test_dataset  = TsDataset(data=test_data,
                              input_size=int(mc['n_time_in']),
                              output_size=int(mc['n_time_out']),
                              sample_freq=int(mc['val_sample_freq']))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=int(mc['batch_size']),
                              shuffle=True,
                              drop_last=True)

    val_loader   = DataLoader(dataset=val_dataset,
                              batch_size=int(mc['eval_batch_size']),
                              shuffle=False)

    test_loader  = DataLoader(dataset=test_dataset,
                              batch_size=int(mc['eval_batch_size']),
                              shuffle=False)

    # ----------------------------------------- Instantiate model ---------------------------------- #
    model = instantiate_model(mc=mc)
    model.training_n_time = train_dataset.data.shape[-1]

    # ----------------------------------------- Fit model ---------------------------------- #
    gpus = -1 if t.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=mc['max_epochs'],
                         max_steps=mc['max_steps'],
                         check_val_every_n_epoch=mc['eval_freq'],
                         gpus=gpus,
                         callbacks=[],
                         logger=False)

    trainer.fit(model, train_loader)

    run_time = time.time() - start

    # ----------------------------------------- Predict ---------------------------------- #
    results = {}

    y_true, y_hat, mask = predict(mc, model, val_loader)
    val_values = (('val_y_true', y_true), ('val_y_hat', y_hat), ('val_mask', mask))
    results.update(val_values)
    
    y_true, y_hat, mask = predict(mc, model, test_loader)
    test_values = (('test_y_true', y_true), ('test_y_hat', y_hat), ('test_mask', mask))
    results.update(test_values)

    # Evaluate in validation set
    val_loss = mae(y=results['val_y_true'], y_hat=results['val_y_hat'], weights=results['val_mask'])

    results_output = {'loss': val_loss,
                      'mc': mc,
                      'run_time': run_time,
                      'status': STATUS_OK}

    forecasts_test = {}
    test_values = (('test_y_true', results['test_y_true']), ('test_y_hat', results['test_y_hat']), ('test_mask', results['test_mask']))
    forecasts_test.update(test_values)
    results_output['forecasts_test'] = forecasts_test

    return results_output

def hyperpar_selection(space,
                       hyperopt_steps,
                       data,
                       ts_val,
                       ts_test,
                       mask,
                       results_file):

    trials = Trials()
    fmin_objective = partial(run_model,
                             data=data, ts_val=ts_val, ts_test=ts_test, mask=mask,
                             trials=trials, results_file=results_file)

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=hyperopt_steps, trials=trials, verbose=True)

    return trials