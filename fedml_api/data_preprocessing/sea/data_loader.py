import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.linear.lr import LogisticRegression


def load_partition_data_sea(batch_size, train_iteration, num_client, drift_together):
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
        change_point = [np.random.random_sample() * train_iteration for c in range(num_client)]

    # Generate data for each client/iteration
    train_data = [[] for t in range(train_iteration + 1)]
    for it in range(train_iteration + 1):
        for c in range(num_client):
            train_df = pd.DataFrame(columns = list(df_con1.columns))
            # Get samples for the first concept
            if it < change_point[c]:
                num_sample = int(min(1.0, change_point[c] - it) * sample_per_client_iter)
                train_df = train_df.append(df_con1.sample(n=num_sample))
            # Get samples for the second concept
            if it + 1 > change_point[c]:
                num_sample = int(min(1.0, it + 1.0 - change_point[c]) * sample_per_client_iter)
                train_df = train_df.append(df_con2.sample(n=num_sample))
            train_data[it].append(train_df)
            # For debugging purposes, save the data as files
            train_df.to_csv(data_path +
                            'client_{}_iter_{}_cp_{}.csv'.format(c, it, change_point[c]),
                            index = False)            
    
    client_num = num_client
    class_num = 2

#    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def main():
    load_partition_data_sea(10, 5, 10, False)
    

if __name__ == '__main__':
    main()
