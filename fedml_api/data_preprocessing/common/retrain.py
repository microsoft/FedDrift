import os
import sys
import json
import pandas as pd
from scipy.stats import poisson

def load_retrain_table_data(data_path, num_client, current_train_iteration,
                            csv_file_name, retrain_method):
    train_data = [pd.DataFrame() for c in range(num_client)]
    test_data = []

    if retrain_method == "all":
        # Use all the data until the current iteration as training data
        for it in range(current_train_iteration + 1):
            for c in range(num_client):
                train_df = pd.read_csv(data_path +
                                       csv_file_name.format(c, it))
                train_data[c] = train_data[c].append(train_df,
                                                     ignore_index=True)
    elif retrain_method.startswith("win-"):
        # Use a window-based approach and only retain data from
        # recent W iterations
        win_size = int(retrain_method.replace("win-", ""))
        start_iter = max(0, current_train_iteration - win_size + 1)
        for it in range(start_iter, current_train_iteration + 1):
            for c in range(num_client):
                train_df = pd.read_csv(data_path +
                                       csv_file_name.format(c, it))
                train_data[c] = train_data[c].append(train_df,
                                                     ignore_index=True)
    elif retrain_method.startswith("weight-"):
        # Use a weighted sampling approach that gives recent data more
        # weights.
        # TODO: We should change the sampling strategy when dealing
        # with a much larger dataset
        weight_method = retrain_method.replace("weight-", "")
        for it in range(current_train_iteration + 1):
            weight = (it+1) if weight_method == "linear" else 2**(it)
            for c in range(num_client):
                train_df = pd.read_csv(data_path +
                                       csv_file_name.format(c, it))
                for w in range(weight):
                    train_data[c] = train_data[c].append(train_df,
                                                         ignore_index=True)

    elif retrain_method.startswith("sel-"):
        # Load data from specific iterations
        select_iter = retrain_method.replace("sel-", "").split(",")
        for it in select_iter:
            itn = int(it)
            for c in range(num_client):
                train_df = pd.read_csv(data_path +
                                       csv_file_name.format(c, itn))
                train_data[c] = train_data[c].append(train_df,
                                                     ignore_index=True)
    elif retrain_method.startswith("clientsel-"):
        # Load data from specific iterations for each client
        train_data_iter = json.loads(retrain_method.replace("clientsel-", ""))
        for c in range(num_client):
            for it in train_data_iter[c]:
                train_df = pd.read_csv(data_path +
                                       csv_file_name.format(c, it))
                train_data[c] = train_data[c].append(train_df,
                                                     ignore_index=True)
    elif retrain_method.startswith("poisson"):
        # Win-1 training, but poisson(1) subsampling
        for c in range(num_client):
            train_df = pd.read_csv(data_path +
                                   csv_file_name.format(c, current_train_iteration))
            weights = poisson.rvs(mu=1, size=train_df.shape[0])
            if sum(weights) != 0:
                train_df = train_df.sample(frac=1, replace=True, weights=weights)
            train_data[c] = train_data[c].append(train_df,
                                                 ignore_index=True)
    else:
        raise NameError(retrain_method)
            
    # Use the data in the next training iteration as the test data
    for c in range(num_client):
        test_df = pd.read_csv(data_path +
                              csv_file_name.format(
                                  c, current_train_iteration + 1))
        test_data.append(test_df)                    

    return train_data, test_data


def load_all_data(data_path, num_client, current_train_iteration, csv_file_name):
    return [ [ pd.read_csv(data_path + csv_file_name.format(c, it)) 
               for it in range(current_train_iteration + 1) ] 
               for c in range(num_client) ]
