"""Preprocess the data: train-test split, normalization, etc."""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import pandas as pd
import array
import csv
from scipy.sparse import csr_matrix, csc_matrix


def load_dataset(data_name,train_ratio=0.5, spinors=False, seed=100):
    """
        spinors: whether to use node-edge-triangle or only edge
    """
    np.random.seed(seed)

    if not spinors:
        if data_name == 'forex':
            with open('data/forex/forex_2018.pkl', 'rb') as f:
                (b1, b2), laplacians, y = pickle.load(f)
                y = y[:,0]

            # for visualization 
        if data_name == 'forex_small':
            with open('data/forex/forex_2018_small.pkl', 'rb') as f:
                (b1, b2), laplacians, y = pickle.load(f)
                y = y[:,0]

        if data_name == 'ocean_flow':
            with open('data/ocean_flow/pacific_data.pkl', 'rb') as f:
                laplacians, eigenvectors, eigenvalues, (b1, b2), _ = pickle.load(f)
            with open('data/ocean_flow/edge_flow.pkl', 'rb') as f:
                y = pickle.load(f)

        n0, n1, n2 = b1.shape[0], b1.shape[1], b2.shape[1]
        num_train = int(train_ratio * n1)
        num_test = n1 - num_train
        x = np.arange(n1)
        random_perm = np.random.permutation(x)
        train_ids, test_ids = random_perm[:num_train], random_perm[num_train:]
        x_train, x_test = x[train_ids], x[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        if data_name not in  ['ocean_flow']:
            return (b1, b2), laplacians, (x_train, y_train), (x_test, y_test), (x, y)
        else: 
            return (b1, b2), laplacians, (x_train, y_train), (x_test, y_test), (x, y), (eigenvectors, eigenvalues)
        
    else:
        if data_name in ['water_network']:
            if data_name == 'water_network':
                with open('data/wsn/water_network.pkl', 'rb') as f:
                    b1, flow_rate, _, head, hr = pickle.load(f)
                    
            hr = hr.squeeze()
            laplacians = (b1@b1.T, b1.T@b1)
            n0, n1 = b1.shape[0], b1.shape[1]
            head = np.array(head)
            flow_rate = np.array(flow_rate)
            sign = np.sign(flow_rate)
            flow_rate = -hr*sign*np.abs(flow_rate)**1.852
            hr[:] = 1
            y = np.concatenate((head, flow_rate))
            
            n = n0+n1
            n0_train = int(train_ratio * n0)
            n1_train = int(train_ratio * n1)
            x0 = np.arange(n0)
            x1 = np.arange(n1)
            x = np.arange(n)
            random_perm0 = np.random.permutation(x0)
            random_perm1 = np.random.permutation(x1)
            train_ids0, test_ids0 = random_perm0[:n0_train], random_perm0[n0_train:]
            # make sure the last node in x0 is in the training set and remove from the test set --- source is typically known
            if n0-1 not in train_ids0:
                train_ids0 = np.concatenate((train_ids0, np.array([n0-1])))
                test_ids0 = np.delete(test_ids0, np.where(test_ids0==n0-1))
                
            train_ids1, test_ids1 = random_perm1[:n1_train], random_perm1[n1_train:]
            # make sure the last edge in x1 is in the training set and remove from the test set --- source is typically known
            if n1-1 not in train_ids1:
                train_ids1 = np.concatenate((train_ids1, np.array([n1-1])))
                test_ids1 = np.delete(test_ids1, np.where(test_ids1==n1-1))
                
            train_ids, test_ids = np.concatenate((train_ids0, train_ids1+n0)), np.concatenate((test_ids0, test_ids1+n0))
            x_train, x_test = x[train_ids], x[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]

            return b1, laplacians, (x_train, y_train), (x_test, y_test), (x, y), (train_ids0, train_ids1), (test_ids0, test_ids1), hr

