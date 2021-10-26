import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import PIL.Image as Image

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.data_preprocessing.common.retrain import load_retrain_table_data, print_change_points, load_all_data
from fedml_api.data_preprocessing.MNIST.data_loader import read_data

def batch_data(data, batch_size):
    '''
    data is in panda frames. the last column is the label
    convert it into batches of (x, y)
    '''
    # randomly shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # convert to numpy arrays
    ndata = data.to_numpy()
    data_x = ndata[:, :-1].astype(np.float64)
    data_y = ndata[:, -1].astype(np.int32)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data

def generate_data_mnist(train_iteration, num_client, drift_together,
                        change_point_str):

    data_path = "./../../../data/MNIST/"

    # We always use 500 samples per client in each training iteration
    # TODO: change to variable sample sizes
    # TODO: generate more clients than requested for client sampling
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
        
    # Read all the MNIST data from file
    mnist = MNIST_Data()
        
    # Generate data for each client/iteration
    train_data = [[] for t in range(train_iteration + 1)]
    for it in range(train_iteration + 1):
        for c in range(num_client):
            train_data = []
            # Get samples for the first concept
            if it < change_point[c]:
                num_sample = int(min(1.0, change_point[c] - it) *
                                 sample_per_client_iter)
                train_data += mnist.generate_sample(num_sample, 0)
            # Get samples for the second concept, which is a 180 deg rotation (k=2)
            if it + 1 > change_point[c]:
                num_sample = int(min(1.0, it + 1.0 - change_point[c]) *
                                 sample_per_client_iter)
                train_data += mnist.generate_sample(num_sample, 2)
            # Save the data as files
            pd.DataFrame(train_data).to_csv(
                data_path + 'client_{}_iter_{}.csv'.format(c, it),
                index = False)
            
    # Write change points for debugging
    with open(data_path + 'change_points', 'w') as cpf:
        for c in range(num_client):
            cpf.write('{}\n'.format(change_point[c]))

def load_partition_data_mnist(batch_size, current_train_iteration,
                              num_client, retrain_data):
    data_path = "./../../../data/MNIST/"

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
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, \
        test_data_global, train_data_local_num_dict, train_data_local_dict, \
        test_data_local_dict, all_data, class_num
        

class MNIST_Data:
    def __init__(self):
        train_path = "./../../../data/MNIST/train"
        test_path = "./../../../data/MNIST/test"
        users, groups, train_data, test_data = read_data(train_path, test_path)

        # aggregate all the data
        X = []
        Y = []
        for u in users:
            X.extend(train_data[u]['x'])
            Y.extend(train_data[u]['y'])
        
        # shuffle the order
        nX = np.asarray(X)
        nY = np.asarray(Y)
        np.random.seed(100)
        rng_state = np.random.get_state()
        np.random.shuffle(nX)
        np.random.set_state(rng_state)
        np.random.shuffle(nY)
        
        self.nX = nX
        self.nY = nY
        self.samples_used = 0

    def generate_sample(self, num_sample, rotation_k):
        samples = []
        
        for i in range(self.samples_used, self.samples_used + num_sample):
            x = self.nX[i]
            y = self.nY[i]
            if rotation_k != 0:
                x2d = np.reshape(x, (28, 28))
                rx2d = np.rot90(x2d, k=rotation_k)
                x = rx2d.flatten()
            samples.append(np.concatenate((x, [y])))
        
        self.samples_used += num_sample
        
        return samples

def main():
    
    mnist = MNIST_Data()

    ex = mnist.nX[0]
    shifted_x = (ex - min(ex))/(max(ex)-min(ex))
    x_2d = np.reshape(shifted_x, (28, 28))
    x_2d_rot = np.rot90(x_2d, k=2)
    img = Image.fromarray(np.uint8(x_2d*255), 'L')
    img_rot = Image.fromarray(np.uint8(x_2d_rot*255), 'L')
    img.show()
    img_rot.show()
    

if __name__ == '__main__':
    main()
