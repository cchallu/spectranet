# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


def instantiate_trend_basis(degree_of_polynomial, size):
        polynomial_size = degree_of_polynomial + 1
        basis = torch.tensor(np.concatenate([np.power(np.arange(size, dtype=float) / size, i)[None, :]
                                             for i in range(polynomial_size)]), dtype=torch.float32)
        return basis


def instantiate_seasonality_basis(harmonics, size):
        frequency = np.append(np.zeros(1, dtype=float), np.arange(harmonics, harmonics / 2 * size, dtype=float) / harmonics)[None, :]

        forecast_grid = 2 * np.pi * ( np.arange(size, dtype=float)[:, None] / size) * frequency

        cos_basis = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
        sin_basis = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
        basis = torch.cat([cos_basis, sin_basis], dim=0)

        return basis


class TrimCenterLayer(nn.Module):
    def __init__(self, output_size):
        """
        Select output_size center along last dimension.
        """
        super(TrimCenterLayer, self).__init__()        
        self.output_size = output_size

    def forward(self, x):
        input_size = x.shape[-1]
        assert input_size >= self.output_size, f'Input size {input_size} is not long enough for {self.output_size}.'

        init = (input_size - self.output_size)//2

        return x[..., init:(init+self.output_size)]


class Generator(nn.Module):
    def __init__(self, window_size, n_filters_multiplier, z_t_dim, n_polynomial, n_harmonics, n_features, max_filters, kernel_size, stride, dilation):
        super(Generator, self).__init__()

        # Basis
        trend_basis = instantiate_trend_basis(degree_of_polynomial=n_polynomial, size=z_t_dim)
        seasonality_basis = instantiate_seasonality_basis(harmonics=n_harmonics, size=z_t_dim)
        self.basis = nn.Parameter(torch.cat([trend_basis, seasonality_basis], dim=0), requires_grad=False)
        self.z_d_dim = len(self.basis)


        n_layers = int(np.log2(window_size/z_t_dim))+1
        layers = []
        filters_list = [self.z_d_dim]
        output_size = z_t_dim

        # Hidden layers
        for i in range(0, n_layers):
            filters = min(max_filters, n_filters_multiplier*(2**(n_layers-i-1)))
            layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=filters, dilation=dilation,
                                             kernel_size=kernel_size, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())

            # Increase temporal dimension from layer1 
            if i > 0:
                output_size *= 2
            layers.append(TrimCenterLayer(output_size=output_size))
            filters_list.append(filters)

        # Output layer
        layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=n_features, dilation=dilation,
                                         kernel_size=kernel_size, stride=stride, padding=1, bias=False))
        layers.append(TrimCenterLayer(output_size=output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        # Basis expansion
        x = x[:,:,None]*self.basis[None, :, :]

        # ConvNet
        for layer in self.layers:
            x = layer(x)

        return x


class _TemporalModel(nn.Module):
    def __init__(self, n_time_in, n_time_out,
                 n_features, univariate, n_layers, n_filters_multiplier, max_filters, kernel_size, stride, dilation,
                 z_t_dim, n_polynomial, n_harmonics, z_iters, z_sigma, z_step_size, z_with_noise, z_persistent,
                 normalize_windows):
        super().__init__()

        # Data
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.window_size = n_time_in+n_time_out
        self.univariate = univariate

        # Generator
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_filters_multiplier = n_filters_multiplier
        self.z_t_dim = z_t_dim
        self.n_polynomial = n_polynomial
        self.n_harmonics = n_harmonics
        self.max_filters = max_filters
        self.kernel_size = kernel_size
        self.stride=stride
        self.dilation=dilation
        self.normalize_windows = normalize_windows

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.univariate:
            n_features = 1

        self.generator = Generator(window_size=self.window_size,
                                   n_features=n_features,
                                   z_t_dim=self.z_t_dim,
                                   n_polynomial=self.n_polynomial,
                                   n_harmonics=self.n_harmonics,
                                   #n_layers=self.n_layers,
                                   n_filters_multiplier=self.n_filters_multiplier,
                                   max_filters=self.max_filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride, dilation=self.dilation)

    def infer_z(self, z, Y, mask, n_iters, with_noise):

        step_size = self.z_step_size
        n_steps = int(np.ceil(n_iters/3))

        # Protection for 0 n_iters
        if n_iters == 0:
            z = torch.autograd.Variable(z, requires_grad=True)
            
        for i in range(n_iters):
            z = torch.autograd.Variable(z, requires_grad=True)
            Y_hat = self.generator(z)
            mse = (Y_hat-Y)**2

            normalizer = torch.sum(mask/len(Y))
            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * (torch.sum(mask*mse)/normalizer) # (Y.shape[1]*Y.shape[2])
            L.backward()
            
            z = z - 0.5 * step_size * (z + z.grad) # self.z_step_size
            if with_noise:
                eps = torch.randn(z.shape).to(z.device)
                z += step_size * eps
            if (i % n_steps == 0) and (i>0):
                step_size = step_size*0.5 

            z = z.detach()

        return z

    def sample_gaussian(self, shape):
        return torch.normal(0, 0.001, shape)

    def _load_current_chain(self, idxs):

        p_0_z = self.p_0_chains[idxs].to(idxs.device)

        if self.univariate:
            p_0_z = p_0_z.reshape(len(p_0_z)*self.n_features,self.generator.z_d_dim)
       
        return p_0_z

    def forward(self, Y, mask, idxs=None, z_0=None):
        
        if self.univariate:
            initial_batch_size, n_features, t_size = Y.shape
            Y = Y.reshape(initial_batch_size*n_features, 1, t_size)
            mask = mask.reshape(initial_batch_size*n_features, 1, t_size)
        
        batch_size = len(Y)

        if self.normalize_windows:
            # Masked mean and std
            sum_mask = torch.sum(mask, dim=2, keepdims=True)
            mask_safe = sum_mask.clone()
            mask_safe[mask_safe==0] = 1

            sum_window = torch.sum(Y, dim=2, keepdims=True)
            mean = sum_window/mask_safe

            sum_square = torch.sum((mask*(Y-mean))**2, dim=2, keepdims=True)
            std = torch.sqrt(sum_square/mask_safe)

            mean[sum_mask==0] = 0.0
            std[sum_mask==0] = 1.0
            std[std==0] = 1.0

            Y = (Y-mean)/std

        if (self.z_persistent) and (idxs is not None) and (z_0 is None):
            z_0 = self._load_current_chain(idxs=idxs)
        elif (z_0 is None):
            z_0 = self.sample_gaussian(shape=(batch_size, self.generator.z_d_dim)).to(Y.device)
        else:
            z_0 = z_0.to(Y.device)

        # Sample z
        z = self.infer_z(z=z_0, Y=Y, mask=mask, n_iters=self.z_iters, with_noise=self.z_with_noise)
        
        # Generator
        Y_hat = self.generator(z)

        if self.normalize_windows:
            Y = Y*std + mean
            Y_hat = Y_hat*std + mean

        if (self.z_persistent) and (idxs is not None):
            if self.univariate:
                z = z.reshape(initial_batch_size, n_features, self.generator.z_d_dim)
                self.p_0_chains[idxs] = z.to(self.p_0_chains.device)
            else:
                self.p_0_chains[idxs] = z.to(self.p_0_chains.device)

        return Y, Y_hat, z


class SpectraNet(pl.LightningModule):
    def __init__(self, n_time_in, n_time_out,
                 n_features, n_layers, n_filters_multiplier, max_filters, kernel_size, stride, dilation,
                 z_t_dim, n_polynomial, n_harmonics, z_iters, z_sigma, z_step_size, z_with_noise, z_persistent, forecasting_mask,
                 normalize_windows, noise_std, learning_rate, lr_decay, lr_decay_step_size, weight_decay, random_seed):
        super(SpectraNet, self).__init__()

        # Generator
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.window_size = n_time_in+n_time_out
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_filters_multiplier = n_filters_multiplier
        self.stride = stride
        self.dilation = dilation
        self.z_t_dim = z_t_dim
        self.n_polynomial = n_polynomial
        self.n_harmonics = n_harmonics
        self.max_filters = max_filters
        self.kernel_size = kernel_size
        self.normalize_windows = normalize_windows
        self.noise_std = noise_std
        self.forecasting_mask = forecasting_mask

        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.weight_decay = weight_decay
        self.random_seed=random_seed

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent

        self.training_n_time = None

        self.univariate = False

        self.model = _TemporalModel(n_time_in=n_time_in, n_time_out=n_time_out, univariate=self.univariate,
                                    n_features=n_features, n_layers=self.n_layers, n_filters_multiplier=n_filters_multiplier, max_filters=max_filters,
                                    kernel_size=kernel_size, stride=self.stride, dilation=self.dilation,
                                    normalize_windows=normalize_windows,
                                    z_t_dim=z_t_dim, n_polynomial=n_polynomial, n_harmonics=n_harmonics, z_iters=z_iters, z_sigma=z_sigma, z_step_size=z_step_size,
                                    z_with_noise=z_with_noise, z_persistent=z_persistent)
        self.z_d_dim = self.model.generator.z_d_dim

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.z_persistent:
            assert self.training_n_time is not None, 'Training length should be defined if z is persistent!'
            if self.univariate:
                self.model.p_0_chains = self.model.sample_gaussian(shape=(self.training_n_time, self.n_features, self.z_d_dim))
            else:
                self.model.p_0_chains = self.model.sample_gaussian(shape=(self.training_n_time, self.z_d_dim))

    def training_step(self, batch, batch_idx):
        Y = batch['Y']
        X = batch['X']
        sample_mask = batch['sample_mask']
        available_mask = batch['available_mask']
        idxs = batch['temporal_idxs']

        # Forecasting
        forecasting_mask = available_mask.clone()
        if (self.forecasting_mask) and (self.n_time_out > 0):
            forecasting_mask[:,:,-self.n_time_out:] = 0

        # Gaussian noise
        Y = Y + self.noise_std*(torch.randn(Y.shape).to(self.device))

        Y, Y_hat, _ = self.model(Y=Y, mask=forecasting_mask, idxs=idxs)

        # Loss
        if self.univariate:
            sample_mask = sample_mask.reshape(Y.shape[0], 1, Y.shape[2])
            available_mask = available_mask.reshape(Y.shape[0], 1, Y.shape[2])

        loss = sample_mask * available_mask * (Y - Y_hat)**2
        loss = torch.mean(loss)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        
        self.lr_schedulers().step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        Y = batch['Y']#.to(self.device)
        X = batch['X']#.to(self.device)
        sample_mask = batch['sample_mask']#.to(self.device)
        available_mask = batch['available_mask']#.to(self.device)
        
        # Forecasting
        forecasting_mask = available_mask.clone()
        if self.n_time_out > 0:
            forecasting_mask[:,:,-self.n_time_out:] = 0

        Y, Y_hat, _ = self.model(Y=Y, mask=forecasting_mask, idxs=None)

        if self.n_time_out > 0:
            Y = Y[:,:,-self.n_time_out:]
            Y_hat = Y_hat[:,:,-self.n_time_out:]
            sample_mask = sample_mask[:,:,-self.n_time_out:]

        # Loss
        loss = sample_mask * available_mask * (Y - Y_hat)**2
        loss = torch.mean(loss)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                 step_size=self.lr_decay_step_size, 
                                                 gamma=self.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
     
    def forward(self, batch, z_0=None):
        Y = batch['Y'].to(self.device)
        X = batch['X'].to(self.device)
        sample_mask = batch['sample_mask'].to(self.device)
        available_mask = batch['available_mask'].to(self.device)
        
        # Forecasting
        forecasting_mask = available_mask.clone()
        if self.n_time_out > 0:
            forecasting_mask[:,:,-self.n_time_out:] = 0

        Y, Y_hat, z  = self.model(Y=Y, mask=forecasting_mask, idxs=None, z_0=z_0)
        
        if self.n_time_out > 0:
            Y = Y[:,:,-self.n_time_out:]
            Y_hat = Y_hat[:,:,-self.n_time_out:]
            sample_mask = sample_mask[:,:,-self.n_time_out:]
        
        return Y, Y_hat, sample_mask, z
