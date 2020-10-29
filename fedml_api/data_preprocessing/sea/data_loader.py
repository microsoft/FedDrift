import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fedml_api.model.linear.lr import LogisticRegression


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_sea(batch_size, train_iteration, num_client, drift_together):
    data_path = "./../../../data/sea/"

    # We always use 1000 samples per client in each training iteration
    # TODO: change to variable sample sizes
    # TODO: generate more clients than requested for client sampling
    sample_per_client_iter = 1000
    
    # For now we only use the first two concepts
    pd_con1 = pd.read_csv(data_path + 'concept1.csv')
    pd_con2 = pd.read_csv(data_path + 'concept2.csv')

    # Randomly generate change point for each client
    change_point = [np.random.random_sample() * train_iteration for c in range(num_client)]

    # Generate data for each client/iteration
    train_data = [[] * (train_iteration + 1)]
    for it in range(train_iteration):
        for c in range(num_client):
            train_df = pd.DataFrame(columns = list(pd_con1.columns))
            # Get samples for the first concept
            if it < change_point[c]:
                num_sample = max(1.0, change_point[c] - it) * sample_per_client_iter
                train_df.append(pd_con1.sample(n=num_sample))
            # Get samples for the second concept
            if it + 1 > change_point[c]:
                num_sample = max(1.0, it + 1.0 - change_point[c]) * sample_per_client_iter
                train_df.append(pd_con2.sample(n=num_sample))
            train_data[it].append(train_df)
            # For debugging purposes, save the data as files
            train_df.to_csv(data_path +
                            'client_{}_iter_{}_cp_{}.csv'.format(c, it, change_point[c]),
                            index = False)
            
    
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def main():
    # test the data loader
    # Hyper Parameters
    input_size = 784
    num_classes = 10
    num_epochs = 50
    batch_size = 10
    learning_rate = 0.03

    np.random.seed(0)
    torch.manual_seed(10)

    device = torch.device("cuda:0")
    client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_mnist(batch_size)

    model = LogisticRegression(input_size, num_classes).to(device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_global):
            images = images.to(device)
            labels = labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if (i + 1) % 100 == 0:
            #     print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
            #           % (epoch + 1, num_epochs, i + 1, len(train_data_global), loss.item()))

        # Test the Model
        correct = 0
        total = 0
        for x, labels in test_data_global:
            x = x.to(device)
            labels = labels.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        # 52% in the last round
        print('Accuracy of the model: %d %%' % (100 * correct // total))


if __name__ == '__main__':
    main()
