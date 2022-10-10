# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pickle
import time
import argparse
import pandas as pd
from hyperopt import hp

from src.hyperpar_selection import hyperpar_selection


def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    n_features, len_dataset = dataset.shape

    interval_size = int(np.ceil(len_dataset/n_intervals))
    mask = np.ones(dataset.shape)
    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u>occlusion_prob)*1
        mask[:, i*interval_size:(i+1)*interval_size] = mask[:, i*interval_size:(i+1)*interval_size]*mask_interval[:,None]

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=1)
    missing_features = np.where(feature_sum==0)[0]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[feature, i*interval_size:(i+1)*interval_size] = 1

    return mask

def get_experiment_space(args):
    if args.horizon == 336:
        n_time_in = [512-args.horizon, 1024-args.horizon]
    elif args.horizon == 192:
        n_time_in = [512-args.horizon, 1024-args.horizon]
    elif args.horizon == 60:
        n_time_in = [128-args.horizon, 256-args.horizon, 512-args.horizon]
    elif args.horizon == 48:
        n_time_in = [128-args.horizon, 256-args.horizon, 512-args.horizon]
    elif args.horizon == 24:
        n_time_in = [128-args.horizon, 256-args.horizon, 512-args.horizon]

    space= {# Architecture parameters
            'n_features': hp.choice('n_features', [ args.n_y ]),
            'n_time_in': hp.choice('n_time_in', n_time_in),
            'n_time_out': hp.choice('n_time_out', [args.horizon]),
            'n_filters_multiplier': hp.choice('n_filters_multiplier', [ 64, 128 ]),
            'max_filters': hp.choice('max_filters', [ 512 ]),
            'n_layers': hp.choice('n_layers', [1]),
            'kernel_size': hp.choice('kernel_size', [4, 8]),
            'stride': hp.choice('stride', [4]),
            'dilation': hp.choice('dilation', [1]),
            'z_t_dim_frac': hp.choice('z_t_dim_frac', [0.5]),
            'n_polynomial': hp.choice('n_polynomial', [ 2 ]),
            'n_harmonics': hp.choice('n_harmonics', [ 1 ]),
            'z_iters': hp.choice('z_iters', [ 25, 35, 50]),
            'z_sigma': hp.choice('z_sigma', [ 0.25 ]),
            'z_step_size': hp.choice('z_step_size', [ 0.05, 0.1, 0.2, 0.3, 0.5]),
            'z_with_noise': hp.choice('z_with_noise', [ None ]),
            'z_persistent': hp.choice('z_persistent', [ True ]),
            'z_iters_inference': hp.choice('z_iters_inference', [ 300, 1000 ]),
            'inference_repeats': hp.choice('inference_repeats', [ 3 ]),
            'forecasting_mask': hp.choice('forecasting_mask', [ False, True ]),
            'normalize_windows': hp.choice('normalize_windows', [ False, True ]),
            'noise_std': hp.choice('noise_std', [ 0.001 ]),
            # Regularization and optimization parameters
            'learning_rate': hp.choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005]),
            'lr_decay': hp.choice('lr_decay', [ 0.5 ] ),
            'n_lr_decays': hp.choice('n_lr_decays', [ 1 ]),
            'weight_decay': hp.choice('weight_decay', [ 0 ] ),
            'max_epochs': hp.choice('max_epochs', [ None ]),
            'max_steps': hp.choice('max_steps', [ 500, 1000 ]),
            'eval_freq': hp.choice('eval_freq', [ 1 ]),
            # Data parameters
            'sample_freq': hp.choice('sample_freq', [1, 24]),
            'val_sample_freq': hp.choice('val_sample_freq', [args.horizon]),
            'batch_size': hp.choice('batch_size', [8, 16, 32]),
            'eval_batch_size': hp.choice('eval_batch_size', [256]),
            'random_seed': hp.quniform('random_seed', 1, 10, 1)}
    return space

def main(args):

    #----------------------------------------------- Load Data -----------------------------------------------#
    data = pd.read_csv(f'./data/{args.dataset}/{args.dataset}.csv')
    
    if args.dataset == 'solar':
        data['hour'] = pd.to_datetime(data['hour'])
        data = data.rename(columns={'hour':'ds'})
        Y = data['y'].values.reshape(32,-1)
        Y = Y/Y.max(axis=1, keepdims=1)
        data['y'] = Y.flatten()
    else:
        data['ds'] = pd.to_datetime(data['ds'])

    data = data.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    print('data: ', data.head())

    if args.dataset in ['simulated7_long', 'simulated7_long_trend', 'simulated7_long_amplitude']:
        ts_val = args.horizon*(1992//args.horizon)
        ts_test = args.horizon*(1992//args.horizon)
        args.n_y = 7
    if args.dataset == 'solar':
        ts_val = args.horizon*(1440//args.horizon)
        ts_test = args.horizon*(1440//args.horizon)
        args.n_y = 32
    if args.dataset == 'ETTm2':
        ts_val = args.horizon*(11520//args.horizon)
        ts_test = args.horizon*(11520//args.horizon)
        args.n_y = 7
    if args.dataset == 'Exchange':
        ts_val = args.horizon*(744//args.horizon)
        ts_test = args.horizon*(1512//args.horizon)
        args.n_y = 8
    if args.dataset == 'Weather':
        ts_val = args.horizon*(5270//args.horizon)
        ts_test = args.horizon*(10536//args.horizon)
        args.n_y = 21
    if args.dataset == 'ILI':
        ts_val =  args.horizon*(96//args.horizon)
        ts_test = args.horizon*(192//args.horizon)
        args.n_y = 7

    space = get_experiment_space(args)

    # Masks
    n_series = args.n_y
    n_time = len(data)//n_series # asssuming square data

    args.occlusion_intervals = int(np.ceil(n_time/args.occlusion_size))

    mask_filename = f'./data/{args.dataset}/mask_{args.occlusion_size}_{args.occlusion_prob}.p'
        
    if os.path.exists(mask_filename):
        print(f'Train mask {mask_filename} loaded!')
        mask = pickle.load(open(mask_filename,'rb'))
    else:
        print('Train mask not found, creating new one')
        mask = get_random_occlusion_mask(dataset=np.ones((n_series, n_time)), n_intervals=args.occlusion_intervals, occlusion_prob=args.occlusion_prob)

        with open(mask_filename,'wb') as f:
            pickle.dump(mask, f)
        print(f'Train mask {mask_filename} created!')

    # Hide data with 0s
    data['y'] = data['y']*mask.flatten()

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/{args.dataset}_{args.occlusion_size}_{args.occlusion_prob}_{args.horizon}/SpectraNet/'

    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    print('Hyperparameter optimization')
    #----------------------------------------------- Hyperopt -----------------------------------------------#
    trials = hyperpar_selection(space=space,
                                hyperopt_steps=args.hyperopt_max_evals,
                                data=data,
                                ts_val=ts_val,
                                ts_test=ts_test,
                                mask=mask,
                                results_file = hyperopt_file)

    with open(hyperopt_file, "wb") as f:
        pickle.dump(trials, f)

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, help='hyperopt_max_evals')
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--occlusion_size', type=int, help='occlusion_intervals')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    occlusion_probs = [0.0, 0.2, 0.4, 0.6, 0.8]
    horizons = [24]
    datasets = [args.dataset]

    # Dataset loop
    for dataset in datasets:
        for horizon in horizons:
            # Occlusion prob loop
            for occlusion_prob in occlusion_probs:
                print(50*'-', dataset, 50*'-')
                print(50*'-', horizon, 50*'-')
                print(50*'-', occlusion_prob, 50*'-')
                start = time.time()
                args.dataset = dataset
                args.horizon = horizon
                args.occlusion_prob = occlusion_prob
                main(args)
                print('Time: ', time.time() - start)
