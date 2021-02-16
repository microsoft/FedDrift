import os
import sys
import pandas as pd

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
            
    # Use the data in the next training iteration as the test data
    for c in range(num_client):
        test_df = pd.read_csv(data_path +
                              csv_file_name.format(
                                  c, current_train_iteration + 1))
        test_data.append(test_df)                    

    return train_data, test_data
    
