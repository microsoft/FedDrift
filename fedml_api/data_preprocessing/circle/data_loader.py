import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.common.retrain import load_retrain_table_data, load_all_data

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

def generate_circle_sample(num_sample, default_concept):
    data_x = np.random.rand(num_sample, 2)
    cr = (0.2, 0.5, 0.15) if default_concept else (0.6, 0.5, 0.25)
    data_y = np.zeros(num_sample)
    z = (data_x[:,0] - cr[0])**2 + (data_x[:,1] - cr[1])**2 - cr[2]**2
    data_y[z>0] = 1

    return np.concatenate((data_x, np.expand_dims(data_y, axis=1)), axis=1)
    
def add_noise(data, noise_prob):
    n = data.shape[0]
    for i in range(n):
        r = np.random.rand(1)[0]
        if r < noise_prob:
            data[i][-1] = 1 - data[i][-1]

def generate_data_circle(train_iteration, num_client, drift_together, sample_per_client_iter,
                         noise_prob, stretch_factor, change_point_str='rand'):

    data_path = "./../../../data/circle/"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Randomly generate a single change point for each client
    if change_point_str == 'rand':
        if drift_together == 1:
            #cp = np.random.random_sample() * train_iteration
            cp = np.random.randint(1, train_iteration//stretch_factor)
            change_point_per_client = [cp for c in range(num_client)]
        else:
            change_point_per_client = [np.random.randint(1, train_iteration//stretch_factor)
                                       for c in range(num_client)]
        
        # matrix of the concept in the training data for each time, client.
        # restricted to concept changes at (time step / stretch) boundary
        change_point = np.zeros((train_iteration//stretch_factor + 1, num_client))
        for c in range(num_client):
            t = change_point_per_client[c]
            change_point[t:,c] = 1
        np.savetxt("./../../../data/changepoints/rand.cp", change_point, fmt='%u')
    
    change_point = np.loadtxt("./../../../data/changepoints/{0}.cp".format(change_point_str), dtype=np.dtype(int))
        
    # Generate data for each client/iteration
    for it in range(train_iteration + 1):
        for c in range(num_client):            
            # train_data = np.array([])            
            # # Get samples for the first concept
            # if it < change_point[c]:
                # num_sample = int(min(1.0, change_point[c] - it) *
                                 # sample_per_client_iter)
                # train_data = generate_circle_sample(num_sample, True)
            # # Get samples for the second concept
            # if it + 1 > change_point[c]:
                # num_sample = int(min(1.0, it + 1.0 - change_point[c]) *
                                 # sample_per_client_iter)
                # new_data = generate_circle_sample(num_sample, False)
                # train_data = np.vstack([train_data, new_data]) \
                             # if train_data.size else new_data
            
            is_default_concept = not(change_point[it//stretch_factor][c])
            train_data = generate_circle_sample(sample_per_client_iter, is_default_concept)                
            add_noise(train_data, noise_prob)
                       
            # Save the data as files
            pd.DataFrame(train_data).to_csv(
                data_path + 'client_{}_iter_{}.csv'.format(c, it),
                index = False, header = ('f1', 'f2', 'label'))
                

def load_all_data_circle(batch_size, current_train_iteration, num_client):
    data_path = "./../../../data/circle/"

    all_data_pd = load_all_data(
        data_path, num_client, current_train_iteration,
        'client_{}_iter_{}.csv')
    
    all_data = list()
    
    for c in range(num_client):        
        all_data_c = list()
        for it in range(current_train_iteration + 1):
            all_data_c.append(batch_data(all_data_pd[c][it], batch_size))
        all_data.append(all_data_c)
        
    return all_data
    

def load_partition_data_circle(batch_size, current_train_iteration,
                             num_client, retrain_data):
    data_path = "./../../../data/circle/"

    # Load the data from generated CSVs
    train_data, test_data = load_retrain_table_data(
        data_path, num_client, current_train_iteration,
        'client_{}_iter_{}.csv', retrain_data)

    
    # Prepare data for FedML
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()

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
    
    client_num = num_client
    class_num = 2

    return client_num, train_data_num, test_data_num, train_data_global, \
        test_data_global, train_data_local_num_dict, train_data_local_dict, \
        test_data_local_dict, class_num


def main():

    np.random.seed(0)
    torch.manual_seed(10)

    generate_data_circle(5, 10, 0, 500, 1.0, 1)

    #generate_data_sea(5, 10, 0)
    
    #client_num, train_data_num, test_data_num, train_data_global, \
    #test_data_global, train_data_local_num_dict, train_data_local_dict, \
    #test_data_local_dict, class_num = \
    #load_partition_data_sea(10, 3, 10)

    #print(client_num)
    #print(train_data_num)
    #print(test_data_num)    
    #print(train_data_local_num_dict)
    #print(class_num)
    #print(test_data_global[0])
    
    

if __name__ == '__main__':
    main()
