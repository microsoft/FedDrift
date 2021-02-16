import logging

import torch
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list, transform_list_to_tensor


class FedAvgEnsTrainer(object):
    def __init__(self, client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, device, models,
                 args):
        self.client_index = client_index
        self.train_data_local_dicts = train_data_local_dicts
        self.train_data_local_num_dicts = train_data_local_num_dicts
        self.all_train_data_nums = train_data_nums

        self.device = device
        self.args = args
        self.models = models
        # logging.info(self.model)        
        self.optimizers = []
        self.criterions = []
        for m in self.models:
            m.to(self.device)
            self.criterions.append(nn.CrossEntropyLoss().to(self.device))
            if self.args.client_optimizer == "sgd":
                self.optimizers.append(torch.optim.SGD(m.parameters(), lr=self.args.lr))
            else:
                self.optimizers.append(torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()),
                                                        lr=self.args.lr,
                                                        weight_decay=self.args.wd, amsgrad=True))

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        for m, w in zip(self.models, weights):
            if self.args.is_mobile == 1:
                w = transform_list_to_tensor(w)

            m.load_state_dict(w)

    def update_dataset(self, client_index):
        self.client_index = client_index

    def train(self):
        results = []

        for mod_idx, model in enumerate(self.models):
            model.to(self.device)
            # change to train mode
            model.train()

            train_local = self.train_data_local_dicts[mod_idx][self.client_index]
            local_sample_number = self.train_data_local_num_dicts[mod_idx][self.client_index]
            criterion = self.criterions[mod_idx]
            optimizer = self.optimizers[mod_idx]

            epoch_loss = []
            for epoch in range(self.args.epochs):
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(train_local):
                    # logging.info(images.shape)
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    logging.info('(client {}, Model {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(
                        self.client_index, mod_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

            weights = model.cpu().state_dict()

            # transform Tensor to list
            if self.args.is_mobile == 1:
                weights = transform_tensor_to_list(weights)
            results.append((weights, local_sample_number))
            
        return results
