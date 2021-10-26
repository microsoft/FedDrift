import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.common.retrain import load_retrain_table_data, print_change_points, load_all_data

def batch_data(data, batch_size):
    '''
    data is in panda frames [f1, f2, label]
    convert it into batches of (x, y)
    '''
    # randomly shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # convert to numpy arrays
    data_x = data[['f1', 'f2']].values.astype(np.float64)
    data_y = data[['label']].values.astype(np.int32).flatten()

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data

def generate_sine_sample(num_sample, default_concept):
    data_x = np.random.rand(num_sample, 2)
    if default_concept:
        data_y = [1 if data_x[i, 1] <= math.sin(data_x[i, 0]) else 0
                  for i in range(num_sample)]
    else:
        data_y = [0 if data_x[i, 1] <= math.sin(data_x[i, 0]) else 1
                  for i in range(num_sample)]
    data_y = np.array(data_y)

    return np.concatenate((data_x, np.expand_dims(data_y, axis=1)), axis=1)

def generate_data_sine(train_iteration, num_client, drift_together,
                       change_point_str):

    data_path = "./../../../data/sine/"

    # We always use 500 samples per client in each training iteration for now
    # TODO: change to variable sample sizes
    # TODO: generate more clients than requested for client sampling
    sample_per_client_iter = 500

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # We always use 500 samples per client in each training iteration
    # TODO: change to variable sample sizes
    
    sample_per_client_iter = 500

    # Randomly generate change point for each client
    if change_point_str == '':
        if drift_together == 1:
            #cp = np.random.random_sample() * train_iteration
            cp = np.random.randint(1, train_iteration)
            change_point = [cp for c in range(num_client)]
        else:
            change_point = [np.random.randint(1, train_iteration)
                            for c in range(num_client)]
    else:
        change_point = json.loads(change_point_str)

    # Print change points
    for idx, cp in enumerate(change_point):
        print('Change point for client {} is {}'.format(idx, cp))
        
    # Generate data for each client/iteration
    train_data = [[] for t in range(train_iteration + 1)]
    for it in range(train_iteration + 1):
        for c in range(num_client):
            train_data = np.array([])
            # Get samples for the first concept
            if it < change_point[c]:
                num_sample = int(min(1.0, change_point[c] - it) *
                                 sample_per_client_iter)
                train_data = generate_sine_sample(num_sample, True)
            # Get samples for the second concept
            if it + 1 > change_point[c]:
                num_sample = int(min(1.0, it + 1.0 - change_point[c]) *
                                 sample_per_client_iter)
                new_data = generate_sine_sample(num_sample, False)
                train_data = np.vstack([train_data, new_data]) \
                             if train_data.size else new_data
                             
            # Save the data as files
            pd.DataFrame(train_data).to_csv(
                data_path + 'client_{}_iter_{}.csv'.format(c, it),
                index = False, header = ('f1', 'f2', 'label'))        
            
    # Write change points for debugging
    with open(data_path + 'change_points', 'w') as cpf:
        for c in range(num_client):
            cpf.write('{}\n'.format(change_point[c]))
    

def load_partition_data_sine(batch_size, current_train_iteration,
                             num_client, retrain_data):
    data_path = "./../../../data/sine/"

    print_change_points(data_path)

    # Load the data from generated CSVs
    train_data, test_data = load_retrain_table_data(
        data_path, num_client, current_train_iteration,
        'client_{}_iter_{}.csv', retrain_data)
    
    all_data_pd = load_all_data(
        data_path, num_client, current_train_iteration,
        'client_{}_iter_{}.csv')
        
    # Prepare data for FedML
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    all_data = list()

    for c in range(num_client):
        train_data_num += len(train_data[c].index)
        test_data_num += len(test_data[c].index)
        train_data_local_num_dict[c] = len(train_data[c].index)

        # transform to batches
        if len(train_data[c].index) > 0:
            train_batch = batch_data(train_data[c], batch_size)
            train_data_local_dict[c] = train_batch
            train_data_global += train_batch
            
        if len(test_data[c].index) > 0:
            test_batch = batch_data(test_data[c], batch_size)        
            test_data_local_dict[c] = test_batch        
            test_data_global += test_batch
            
        all_data_c = list()
        for it in range(current_train_iteration + 1):
            all_data_c.append(batch_data(all_data_pd[c][it], batch_size))
        all_data.append(all_data_c)
    
    client_num = num_client
    class_num = 2

    return client_num, train_data_num, test_data_num, train_data_global, \
        test_data_global, train_data_local_num_dict, train_data_local_dict, \
        test_data_local_dict, all_data, class_num


def main():

    np.random.seed(0)
    torch.manual_seed(10)

    generate_data_sine(5, 10, 0)

    #generate_data_sea(5, 10, 0)
    
    #client_num, train_data_num, test_data_num, train_data_global, \
    #test_data_global, train_data_local_num_dict, train_data_local_dict, \
    #test_data_local_dict, all_data, class_num = \
    #load_partition_data_sea(10, 3, 10)

    #print(client_num)
    #print(train_data_num)
    #print(test_data_num)    
    #print(train_data_local_num_dict)
    #print(class_num)
    #print(test_data_global[0])
    
    

if __name__ == '__main__':
    main()
