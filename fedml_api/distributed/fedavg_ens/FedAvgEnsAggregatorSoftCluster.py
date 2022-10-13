import copy
import logging
import time

import torch
import wandb
import numpy as np
import pickle
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor, transform_tensor_to_list
from fedml_api.distributed.fedavg_ens.FedAvgEnsDataLoader import SoftClusterState
from fedml_api.model.utils import reinitialize


class FedAvgEnsAggregatorSoftCluster(object):
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
        self.sc_state = self.init_sc_state()

    def init_model(self, models):
        for m in models:
            model_params = m.state_dict()
        # logging.info(model)
        return models

    def init_sc_state(self):
        # Load the previous state
        with open('sc_state.pkl', 'rb') as f:
            sc_state = pickle.load(f)
            
        # Run the clustering at the beginning of the iteration
        # assume 1 concept at first iteration
        
        if 'H' in self.args.concept_drift_algo_arg:
            if self.args.curr_train_iteration == 0:
                sc_state.cluster_init()
            else:
                sc_state.cluster_hierarchical(self.args.curr_train_iteration, self.models, self.all_data, self.device)
        elif 'cfl' in self.args.concept_drift_algo_arg:
            if self.args.curr_train_iteration == 0:
                sc_state.cluster_init()
            else:
                sc_state.cluster_cfl_init(self.args.curr_train_iteration)
        elif 'hard' in self.args.concept_drift_algo_arg:
            if self.args.curr_train_iteration == 0:
                for model in self.models:
                    for layer in model.children():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
            curr_acc = self.train_acc_matrix()
            sc_state.cluster(curr_acc, self.args.curr_train_iteration, 0)
        elif 'mmacc' in self.args.concept_drift_algo_arg:
            if self.args.curr_train_iteration == 0:
                sc_state.cluster_init()
            else:
                sc_state.cluster_mmacc2(self.args.curr_train_iteration, self.models, self.all_data, self.device)
        else:
            # hardcoded type D, which has multiple concepts at time 0
            if self.args.curr_train_iteration == 0:
                sc_state.cluster_init()
            else:          
                curr_acc = self.train_acc_matrix()
                
                # If reset variant, delete models that are not epsilon-better than the rest
                if self.args.concept_drift_algo == 'softclusterreset':
                    # if all(curr_acc[1] < curr_acc[0] + 0.01):     # simple case for 2 models
                    deleted_models = []
                    for m in reversed(range(len(self.models))):
                        rest = np.delete(curr_acc, deleted_models+[m], axis=0)
                        if rest.shape[0] > 0:
                            if all(curr_acc[m] < np.max(rest, axis=0) + 0.01):
                                deleted_models.append(m)
                                wandb.run.summary["Reset-{}".format(m)] = 1
                                sc_state.set_weights_zero_model(m)
                                reinitialize(self.models[m])
                    if len(deleted_models) > 0:
                        curr_acc = self.train_acc_matrix()
                
                # Cluster on current accuracy
                sc_state.cluster(curr_acc, self.args.curr_train_iteration, 0)
                
        # If win-1 variant, reset weights of prev iters
        if self.args.concept_drift_algo == 'softclusterwin-1':
            sc_state.set_weights_win1(self.args.curr_train_iteration)
        
        # record accuracy at first iteration so that drift detector is initialized
        if self.args.curr_train_iteration == 0:       
            for client_idx in range(self.args.client_num_in_total):
                test_model_idx = sc_state.get_test_model_idx(self.args.curr_train_iteration, client_idx)
                train_data = self.all_data[client_idx][self.args.curr_train_iteration]
                train_tot_correct, train_num_sample, train_loss = self._infer(self.models[test_model_idx],
                                                                              train_data)
                train_acc = 0
                if train_num_sample != 0:
                    train_acc = train_tot_correct/train_num_sample
                sc_state.set_acc(client_idx, train_acc)
        
        return sc_state

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
        
    def aggregate(self, round_idx):
        start_time = time.time()
        
        if 'cfl' in self.args.concept_drift_algo_arg:
            did_split = self.sc_state.cluster_cfl(self.args.curr_train_iteration, round_idx+1, self.models, self.weights_and_num_samples_dict)
            if did_split:   
                # skip this round, since the local updates correspond to an outdated set of models
                end_time = time.time()
                logging.info("aggregate time cost: %d" % (end_time - start_time))
                return self.get_global_model_params()

        # Do aggregate for all models one by one
        for m_idx in range(len(self.models)):
            
            # For efficiency, don't train a model that is not being used at the current iteration
            if not any( self.sc_state.get_weights()[self.args.curr_train_iteration][m_idx] ):
                continue
            
            model_list = []
            total_weight = 0

            for idx in range(self.worker_num):
                # num_sample returned by the client already factors in the sc_state weights
                model, num_sample = self.weights_and_num_samples_dict[idx][m_idx]
                if num_sample > 0:
                    if self.args.is_mobile == 1:
                        model = transform_list_to_tensor(model)
                    model_list.append((num_sample, model))
                    total_weight += num_sample
            
            # Skip the model that has no updates from any client
            if total_weight == 0:
                continue
            
            #logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

            # logging.info("################aggregate: %d" % len(model_list))
            (num0, averaged_params) = model_list[0]
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_weight, local_model_params = model_list[i]
                    w = local_weight / total_weight
                    if i == 0:                        
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # update the global model which is cached at the server side
            self.models[m_idx].load_state_dict(averaged_params)
        
        if self.args.concept_drift_algo_arg == "hard-r":
            # update the clustering every round
            curr_acc = self.train_acc_matrix()
            # inflate round_idx by 1 to distinguish from the initial clustering
            self.sc_state.cluster(curr_acc, self.args.curr_train_iteration, round_idx+1)
            
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
        return {'sc_weights': self.sc_state.get_weights()}

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
                test_model_idx = self.sc_state.get_test_model_idx(self.args.curr_train_iteration, client_idx)
                train_model_idx = test_model_idx
                
                # train data (for only the current iter)
                train_data = self.all_data[client_idx][self.args.curr_train_iteration]
                train_tot_correct, train_num_sample, train_loss = self._infer(self.models[train_model_idx],
                                                                              train_data)
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
                
                # # test data over specific digits
                # if self.args.dataset == 'MNIST':
                    # test_acc6 = self._infer_class(self.models[test_model_idx],
                                                  # self.test_data_local_dicts[test_model_idx][client_idx],
                                                  # 6)
                    # test_acc9 = self._infer_class(self.models[test_model_idx],
                                                  # self.test_data_local_dicts[test_model_idx][client_idx],
                                                  # 9)
                    # if self.args.report_client == 1:
                        # wandb.log({"Test/Digit6-CL-{}".format(client_idx): test_acc6,
                                   # "round": round_idx})
                        # wandb.log({"Test/Digit9-CL-{}".format(client_idx): test_acc9,
                                   # "round": round_idx})
                    

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

        # Save SC state
        if round_idx > (self.args.comm_round - 5):
            with open('sc_state.pkl','wb') as f:
                pickle.dump(self.sc_state, f)
                
    def reported_acc(self, correct, num_sample):
        if num_sample == 0:
            return -1
        else:
            return correct/num_sample
                
    def train_acc_matrix(self):
        acc = np.zeros((len(self.models), self.args.client_num_in_total))
        
        for m in range(len(self.models)):
            for c in range(self.args.client_num_in_total):
                data = self.all_data[c][self.args.curr_train_iteration]
                num_correct, num_sample, _ = self._infer(self.models[m], data)
                if num_sample != 0:
                    acc[m][c] = num_correct/num_sample
        
        return acc

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
     
    # accuracy of model on only the test_data with the class_label
    def _infer_class(self, model, test_data, class_label):
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
                
                correct_pred = predicted.eq(target)
                index_class = torch.where(target == class_label, True, False)
                correct_pred_class = torch.masked_select(correct_pred, index_class)
                
                test_acc += correct_pred_class.sum().item()
                test_total += index_class.sum().item()

        model.to(torch.device('cpu'))
        
        if test_total == 0:
            return -1
        else:
            return test_acc/test_total
