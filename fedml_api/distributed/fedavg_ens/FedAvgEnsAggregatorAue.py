import copy
import logging
import time

import torch
import wandb
import numpy as np
import pickle
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor


class FedAvgEnsAggregatorAue(object):
    EPS = 1e-20
    K = 5
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                 worker_num, device, model, prev_models, class_num, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.prev_models = prev_models
        self.class_num = class_num
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)
        self.ens_weights = self.init_ens_weights()
        logging.info("Ensemble weights==>")
        logging.info(self.ens_weights)

    def init_model(self, model):
        model_params = model.state_dict()
        # logging.info(model)
        return model, model_params

    def init_ens_weights(self):
        # Initialize the ensemble weights
        # The equation is based on the AUE papar
        py = 1.0/self.class_num
        mser = ((1-py)**2)
        weights = []
        # Calculate MSE for each previous model based on the
        # most recent batch of data. We assume the training
        # data here is the most recent batch (e.g., no data
        # from previous training iterations)
        for m in self.prev_models:
            msei = total_sample = 0.
            for client_idx in range(self.args.client_num_in_total):
                mse, sample = self._mse(
                    m, self.train_data_local_dict[client_idx])
                msei += mse
                total_sample += sample
            msei = msei/total_sample
            weights.append(1./(mser + msei + FedAvgEnsAggregatorAue.EPS))

        # The most recent classifier got the "perfect" weighting
        weights.append(1./(mser + FedAvgEnsAggregatorAue.EPS))
        ens_weight = np.array(weights)

        # Remove the classifier that has the smallest weight
        if len(weights) > FedAvgEnsAggregatorAue.K:
            rm_idx = np.argmin(ens_weight[:-1])
            ens_weight = np.delete(ens_weight, [rm_idx])
            self.prev_models.pop(rm_idx)
        
        return ens_weight/ens_weight.sum()  #normalize            

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
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
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

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
        self.model.load_state_dict(averaged_params)

        # Save the models if we are reaching the end of training
        if round_idx > (self.args.comm_round - 10):
            model_list = []
            for model in self.prev_models:
                model_list.append(copy.deepcopy(model).cpu())
            model_list.append(copy.deepcopy(self.model).cpu())
            with open('model_iter_{}.pkl'.format(
                    self.args.curr_train_iteration), 'wb') as f:
                pickle.dump(model_list, f)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            test_num_samples = []
            test_tot_corrects = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                train_tot_correct, train_num_sample, train_loss = self._infer(self.train_data_local_dict[client_idx])
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # test data
                test_tot_correct, test_num_sample = self._infer_ens(self.test_data_local_dict[client_idx])
                test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                test_num_samples.append(copy.deepcopy(test_num_sample))

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
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            stats = {'test_acc': test_acc}
            logging.info(stats)

    def _mse(self, model, test_data):
        model.eval()
        model.to(self.device)
        mse = test_total = 0.
        softmax = nn.Softmax(dim=1).to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                pred = model(x)
                prob = softmax(pred)
                prob_np = prob.detach().cpu().numpy()
                pr = prob_np[np.arange(target.size(0)),
                             copy.deepcopy(target).detach().cpu().numpy()]
                mse += ((1.0-pr)**2).sum()
                test_total += target.size(0)
        return mse, test_total

    def _infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss

    def _infer_ens(self, test_data):
        # Set up the model list
        model_list = self.prev_models.copy()
        model_list.append(self.model)
        for model in model_list:
            model.eval()
            model.to(self.device)

        test_acc = test_total = 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                pred_pr = np.zeros((target.size(0), self.class_num))
                x = x.to(self.device)
                target = target.to(self.device)
                # Go through all models
                for model, weight in zip(model_list, self.ens_weights):
                    pred = model(x)                    
                    _, predicted = torch.max(pred, -1)
                    predicted = predicted.detach().cpu().numpy()
                    pred_pr[np.arange(target.size(0)), predicted] += weight
                # Get the overall prediction
                overall_pred = np.argmax(pred_pr, -1)
                target_np = copy.deepcopy(target).detach().cpu().numpy()
                correct = (overall_pred == target_np).sum()
                test_acc += correct
                test_total += target.size(0)

        return test_acc, test_total
