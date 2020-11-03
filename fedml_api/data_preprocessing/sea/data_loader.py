import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.linear.lr import LogisticRegression

def batch_data(data, batch_size):
    '''
    data is in panda frames [f1, f2, f3, label]
    convert it into batches of (x, y)
    '''
    # randomly shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # convert to numpy arrays
    data_x = data[['f1', 'f2', 'f3']].values.astype(np.float64)
    data_y = data[['label']].values.astype(np.int32)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data
    

def load_partition_data_sea(batch_size, train_iteration, num_client,
                            drift_together):
    data_path = "./../../../data/sea/"

    # We always use 500 samples per client in each training iteration
    # TODO: change to variable sample sizes
    # TODO: generate more clients than requested for client sampling
    sample_per_client_iter = 500
    
    # For now we only use the first two concepts
    df_con1 = pd.read_csv(data_path + 'concept1.csv')
    df_con2 = pd.read_csv(data_path + 'concept2.csv')

    # Randomly generate change point for each client
    if drift_together:
        cp = np.random.random_sample() * train_iteration
        change_point = [cp for c in range(num_client)]
    else:
        change_point = [np.random.random_sample() * train_iteration
                        for c in range(num_client)]

    # Generate data for each client/iteration
    train_data = [[] for t in range(train_iteration + 1)]
    for it in range(train_iteration + 1):
        for c in range(num_client):
            train_df = pd.DataFrame(columns = list(df_con1.columns))
            # Get samples for the first concept
            if it < change_point[c]:
                num_sample = int(min(1.0, change_point[c] - it) *
                                 sample_per_client_iter)
                train_df = train_df.append(df_con1.sample(n=num_sample),
                                           ignore_index=True)
            # Get samples for the second concept
            if it + 1 > change_point[c]:
                num_sample = int(min(1.0, it + 1.0 - change_point[c]) *
                                 sample_per_client_iter)
                train_df = train_df.append(df_con2.sample(n=num_sample),
                                           ignore_index=True)
            train_data[it].append(train_df)
            # For debugging purposes, save the data as files
            train_df.to_csv(data_path +
                            'client_{}_iter_{}.csv'.format(c, it),
                            index = False)
            
    # Write change points for debugging
    with open(data_path + 'change_points', 'w') as cpf:
        for c in range(num_client):
            cpf.write('{}\n'.format(change_point[c]))

    # Prepare data into multiple training iterations
    train_data_num = [0 for t in range(train_iteration)]
    test_data_num = [0 for t in range(train_iteration)]
    train_data_local_dict = [dict() for t in range(train_iteration)]
    test_data_local_dict = [dict() for t in range(train_iteration)]
    train_data_local_num_dict = [dict() for t in range(train_iteration)]
    train_data_global = [list() for t in range(train_iteration)]
    test_data_global = [list() for t in range(train_iteration)]

    for t in range(train_iteration):
        for c in range(num_client):
            # We use all the data before the current iteration
            # as training data
            # TODO: change it to an option for other methods
            train_df = pd.DataFrame(columns = list(train_data[0][c].columns))
            for it in range(t+1):
                train_df = train_df.append(train_data[it][c],
                                           ignore_index=True)
            train_data_num[t] += len(train_df.index)
            # test data is from the next training iteration
            test_data_num[t] += len(train_data[t+1][c].index)
            train_data_local_num_dict[t][c] = len(train_df.index)

            # transform to batches
            train_batch = batch_data(train_df, batch_size)
            test_batch = batch_data(train_data[t+1][c], batch_size)

            # put batched data into the arrays
            train_data_local_dict[t][c] = train_batch
            test_data_local_dict[t][c] = test_batch
            
            train_data_global[t] += train_batch
            test_data_global[t] += test_batch
    
    client_num = num_client
    class_num = 2

    return client_num, train_data_num, test_data_num, train_data_global, \
        test_data_global, train_data_local_num_dict, train_data_local_dict, \
        test_data_local_dict, class_num


def main():
    client_num, train_data_num, test_data_num, train_data_global, \
    test_data_global, train_data_local_num_dict, train_data_local_dict, \
    test_data_local_dict, class_num = \
    load_partition_data_sea(10, 5, 10, False)

    print(client_num)
    print(train_data_num)
    print(test_data_num)    
    print(train_data_local_num_dict)
    print(class_num)
    
    

if __name__ == '__main__':
    main()
