import logging

import torch
from torch import nn
import numpy as np

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list


class FedAVGTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 args):
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):
        # Skip the training if there is no training data for this model
        if self.local_sample_number == 0:
            return None, 0
            
        self.model.to(self.device)
        # change to train mode
        self.model.train()
        
        if isinstance(self.train_local, list):
            for step in range(self.args.epochs):
                batch_idx = np.random.choice(len(self.train_local))
                (x, labels) = self.train_local[batch_idx]
                
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
        elif isinstance(self.train_local, torch.utils.data.dataloader.DataLoader):
            for step in range(self.args.epochs):
                (x, labels) = next(iter(self.train_local))
                
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
