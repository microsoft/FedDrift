import copy
import logging
import time

import torch
import wandb
import numpy as np
import pickle
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor, transform_tensor_to_list
from fedml_api.distributed.fedavg_ens.FedAvgEnsDataLoader import AdaState


class FedAvgEnsAggregatorAda(object):
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
        self.ada_state = self.init_ada_state()
        
        if args.concept_drift_algo_arg.split('_')[1] == 'round':
            self.update_each_round = True
        elif args.concept_drift_algo_arg.split('_')[1] == 'iter':
            self.update_each_round = False
        else:
            raise NameError('ada config')
        

    def init_model(self, models):
        for m in models:
            model_params = m.state_dict()
        # logging.info(model)
        return models

    def init_ada_state(self):
        # Load the previous state
        with open('ada_state.pkl', 'rb') as f:
            ada_state = pickle.load(f)
        return ada_state

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

        # Do aggregate for all models one by one
        for m_idx in range(len(self.models)):
            model_list = []
            training_num = 0

            for idx in range(self.worker_num):
                model, num_sample = self.weights_and_num_samples_dict[idx][m_idx]
                if num_sample > 0:
                    if self.args.is_mobile == 1:
                        model = transform_list_to_tensor(model)
                    model_list.append((num_sample, model))
                    training_num += num_sample

            #logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

            # logging.info("################aggregate: %d" % len(model_list))
            (num0, averaged_params) = model_list[0]
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # update the global model which is cached at the server side
            self.models[m_idx].load_state_dict(averaged_params)
            
            
            # update the learning rate
            theta = np.array([])
            for (k,v) in averaged_params.items():
                theta = np.append(theta, v.numpy().flatten())
            
            if self.update_each_round:
                t = round_idx + self.args.curr_train_iteration*self.args.comm_round
                self.ada_state.update(theta, t)
            else:
                if round_idx == self.args.comm_round - 5:
                    self.ada_state.update(theta, self.args.curr_train_iteration)
            
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
        return {'lr': self.ada_state.current_lr()}

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
                train_model_idx = 0
                test_model_idx = 0
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
                    wandb.log({"Train/Acc-CL-{}".format(client_idx): train_tot_correct/train_num_sample,
                               "round": round_idx})
                    wandb.log({"Test/Acc-CL-{}".format(client_idx): test_tot_correct/test_num_sample,
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

        # Save ada state
        if round_idx > (self.args.comm_round - 5):
            with open('ada_state.pkl','wb') as f:
                pickle.dump(self.ada_state, f)


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

        return test_acc, test_total, test_loss
