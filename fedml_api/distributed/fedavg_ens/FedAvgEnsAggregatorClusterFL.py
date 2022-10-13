import copy
import logging
import time

import torch
import wandb
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor, transform_tensor_to_list


class FedAvgEnsAggregatorClusterFL(object):
    def __init__(self, train_globals, test_globals, all_train_data_nums,
                 train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts,
                 all_data, worker_num, device, models, class_num, args):
        self.train_globals = train_globals
        self.test_globals = test_globals
        self.all_train_data_nums = all_train_data_nums
        self.all_data = all_data

        self.train_data_local_dicts = train_data_local_dicts
        self.test_data_local_dicts = test_data_local_dicts
        self.train_data_local_num_dicts = train_data_local_num_dicts

        self.worker_num = worker_num
        self.device = device
        self.class_num = class_num
        self.args = args
        self.weights_and_num_samples_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.models = self.init_model(models)
        self.is_split = False
        self.cluster_indices = [np.arange(self.args.client_num_in_total)]
        self.cluster_assignment = [0 for i in range(self.args.client_num_in_total)]
        self.max_eps_1 = 0
        self.EPS_1 = 0
        self.EPS_2 = 10000

    def init_model(self, models):
        for m in models:
            model_params = m.state_dict()
        # logging.info(model)
        return models

    def get_global_model_params(self):
        model_params = [model.state_dict() for model in self.models]
        return model_params

    def add_local_trained_result(self, index, weights_and_num_samples):
        logging.info("add_model. index = %d" % index)
        self.weights_and_num_samples_dict[index] = weights_and_num_samples
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def compute_weight_update(self):
        weight_updates = []
        for c_idx in range(self.worker_num):
            # Find the first model that has data. We assume one client can only
            # contribute to one model
            for m_idx in range(len(self.models)):
                model, num_sample = self.weights_and_num_samples_dict[c_idx][m_idx]
                if num_sample <= 0:
                    continue
                # Load the original model
                old_model = self.models[m_idx].cpu().state_dict()
                diff = dict()
                for k in old_model.keys():
                    diff[k] = model[k] - old_model[k]
                weight_updates.append(diff)
                break
        return weight_updates

    def flatten(self, source):
        return np.concatenate([value.flatten() for value in source.values()],
                              axis = None)

    def compute_max_update_norm(self, weight_updates):
        return np.max([np.linalg.norm(self.flatten(dW))
                       for dW in weight_updates])

    
    def compute_mean_update_norm(self, weight_updates):
        # TODO: Consider the sample size when doing the averaging
        return np.linalg.norm(np.mean(np.stack([self.flatten(dW) for dW in weight_updates]), axis=0))

    def compute_pairwise_similarities(self, weight_updates):
        angles = np.zeros((len(weight_updates), len(weight_updates)))
        for i, source1 in enumerate(weight_updates):
            for j, source2 in enumerate(weight_updates):
                s1 = self.flatten(source1)
                s2 = self.flatten(source2)
                angles[i,j] = np.dot(s1,s2)/(np.linalg.norm(s1)*np.linalg.norm(s2)+1e-12)

        return angles

    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        return c1, c2

    def aggregate(self, round_idx):
        start_time = time.time()

        # Check if we need to split if we haven't (we only split once for now)
        # TODO: Make it recursive like the original paper
        if not self.is_split:
            weight_updates = self.compute_weight_update()
            max_norm = self.compute_max_update_norm(weight_updates)
            mean_norm = self.compute_mean_update_norm(weight_updates)
            wandb.log({"Max_Norm": max_norm, "round": round_idx})
            wandb.log({"Mean_Norm": mean_norm, "round": round_idx})
            mean_norm_increase = False
            if mean_norm > self.max_eps_1:
                self.max_eps_1 = mean_norm
                mean_norm_increase = True
                # Set EPS_1 as 1/10 of the max mean norm
                self.EPS_1 = self.max_eps_1/10.0
                # Set EPS_2 as 6 * EPS_1
                self.EPS_2 = 6 * self.EPS_1
                logging.info("Round {}, Set EPS_1 = {}, EPS_2 = {}".format(
                    round_idx, self.EPS_1, self.EPS_2))
            if mean_norm < self.EPS_1 and max_norm > self.EPS_2 and \
               round_idx > 100 and (not mean_norm_increase):
                similarities = self.compute_pairwise_similarities(weight_updates)
                c1, c2 = self.cluster_clients(similarities)
                self.cluster_indices = [c1, c2]
                for cl_idx, cl in enumerate(self.cluster_indices):
                    for cc in cl:
                        self.cluster_assignment[cc] = cl_idx
                self.is_split = True
                print("splitting clusters at round {}".format(round_idx))
                print(self.cluster_indices)
                print(self.cluster_assignment)

        # Do aggregate for all models one by one
        for m_idx in range(len(self.models)):
            model_list = []
            training_num = 0

            if m_idx >= len(self.cluster_indices):
                # Stop aggregating if the number of models is larger than
                # the number of clusters
                break
            for c_idx in self.cluster_indices[m_idx]:
                # Find the contribution of this client (may not match the
                # the model index because the clustering changes)
                for mm in range(len(self.models)):
                    model, num_sample = self.weights_and_num_samples_dict[c_idx][mm]
                    if num_sample <= 0:
                        continue
                    if self.args.is_mobile == 1:
                        model = transform_list_to_tensor(model)
                    model_list.append((num_sample, model))
                    training_num += num_sample
                    break
                
            # logging.info("################aggregate: %d" % len(model_list))
            (num0, averaged_params) = model_list[0]
            for k in averaged_params.keys():
                init = False
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if not init:                        
                        averaged_params[k] = local_model_params[k] * w
                        init = True
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # update the global model which is cached at the server side
            self.models[m_idx].load_state_dict(averaged_params)
            
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return self.get_global_model_params()

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def extra_info(self, round_idx):
        return self.cluster_assignment

    def test_on_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            test_num_samples = []
            test_tot_corrects = []
            test_losses = []
            for client_idx in range(self.args.client_num_in_total):
                train_model_idx = test_model_idx = self.cluster_assignment[client_idx]
                # train data
                train_tot_correct, train_num_sample, train_loss = self._infer(self.models[train_model_idx],
                                                                              self.train_data_local_dicts[train_model_idx][client_idx])
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # test data
                test_tot_correct, test_num_sample, test_loss = self._infer(self.models[test_model_idx],
                                                                           self.test_data_local_dicts[test_model_idx][client_idx])
                test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                test_num_samples.append(copy.deepcopy(test_num_sample))
                test_losses.append(copy.deepcopy(test_loss))
                if self.args.report_client == 1:
                    wandb.log({"Train/Acc-CL-{}".format(client_idx): self.reported_acc(train_tot_correct, train_num_sample),
                               "round": round_idx})
                    wandb.log({"Test/Acc-CL-{}".format(client_idx): self.reported_acc(test_tot_correct, test_num_sample),
                               "round": round_idx})

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

    def reported_acc(self, correct, num_sample):
        if num_sample == 0:
            return -1
        else:
            return correct/num_sample

    def _infer(self, model, test_data):
        model.eval()
        model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        model.to(torch.device('cpu'))

        return test_acc, test_total, test_loss
