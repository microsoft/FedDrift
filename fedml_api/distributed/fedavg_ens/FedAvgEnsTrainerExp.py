import logging

import torch
from torch import nn
import numpy as np

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list, transform_list_to_tensor


class FedAvgEnsTrainerExp(object):
    def __init__(self, client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data, device, models,
                 args):
        self.client_index = client_index
        self.train_data_local_dicts = train_data_local_dicts
        self.train_data_local_num_dicts = train_data_local_num_dicts
        self.all_train_data_nums = train_data_nums
        self.all_local_data = all_local_data

        self.device = device
        self.args = args
        self.models = models
        # logging.info(self.model)        
        self.optimizers = []
        self.criterions = []
        for m in self.models:
            #m.to(self.device)
            self.criterions.append(nn.CrossEntropyLoss().to(self.device))
            if self.args.client_optimizer == "sgd":
                self.optimizers.append(torch.optim.SGD(m.parameters(), lr=self.args.lr))
            else:
                self.optimizers.append(torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()),
                                                        lr=self.args.lr,
                                                        weight_decay=self.args.wd, amsgrad=True))

    def update_model(self, weights, extra_info):
        # logging.info("update_model. client_index = %d" % self.client_index)
        for m, w in zip(self.models, weights):
            if self.args.is_mobile == 1:
                w = transform_list_to_tensor(w)

            m.load_state_dict(w)
        self.extra_info = extra_info

    def update_dataset(self, client_index):
        self.client_index = client_index

    def train(self):
        results = {}

        for mod_idx, model in enumerate(self.models):
            model.to(self.device)
            # change to train mode
            model.train()

            local_sample_number = sum( len(self.all_local_data[t]) 
                                       for t in range(len(self.all_local_data)) )
            
            # Skip the training if there is no training data for this model
            if local_sample_number == 0:
                results[mod_idx] = (None, 0)
                continue

            criterion = self.criterions[mod_idx]
            optimizer = self.optimizers[mod_idx]
            
            unnorm_probs = np.asarray([ 2**t for t in range(len(self.all_local_data)) ])
            probs = unnorm_probs/sum(unnorm_probs)

            for step in range(self.args.epochs):
                t_sample = np.random.choice(len(self.all_local_data), p=probs)
                data_t = self.all_local_data[t_sample]
                if len(data_t) == 0:
                    continue
                    
                if isinstance(data_t, list):
                    batch_idx = np.random.choice(len(data_t))
                    (x, labels) = data_t[batch_idx]
                elif isinstance(data_t, torch.utils.data.dataloader.DataLoader):
                    (x, labels) = next(iter(data_t))
                
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

            model.to(torch.device('cpu'))
            weights = model.state_dict()

            # transform Tensor to list
            if self.args.is_mobile == 1:
                weights = transform_tensor_to_list(weights)
            results[mod_idx] = (weights, local_sample_number)
            
        return results
