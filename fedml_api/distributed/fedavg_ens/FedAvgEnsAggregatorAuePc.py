import copy
import logging
import time

import torch
import wandb
import numpy as np
import pickle
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_list_to_tensor, transform_tensor_to_list


class FedAvgEnsAggregatorAuePc(object):
    EPS = 1e-20
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
        self.ens_weights = self.init_ens_weights()        

    def init_model(self, models):
        for m in models:
            model_params = m.state_dict()
        # logging.info(model)
        return models

    def init_ens_weights(self):
        # Initialize the ensemble weights
        # At the beginning we assign all models to be "perfect"
        all_client_weights = []        
        py = 1.0/self.class_num
        mser = ((1-py)**2)
        for client_idx in range(self.args.client_num_in_total):
            ens_weights = np.full(len(self.models),
                                  1./(mser + FedAvgEnsAggregatorAuePc.EPS))
            #normalize
            all_client_weights.append(ens_weights/ens_weights.sum())
        return all_client_weights
    

    def update_ens_weights(self):
        # Initialize the ensemble weights
        # The equation is based on the AUE papar
        py = 1.0/self.class_num
        mser = ((1-py)**2)

        # Calculate MSE for each previous model based on the
        # most recent batch of data. We assume the training
        # data here is the most recent batch (e.g., no data
        # from previous training iterations)
        # Here, we calculate weights for each client
        for client_idx in range(self.args.client_num_in_total):
            for m_idx, model in enumerate(self.models[1:]):
                # Always use the most recent batch for MSE
                mse, sample = self._mse(
                    model, self.train_data_local_dicts[0][client_idx])
                if sample == 0:
                    msei = 0
                else:
                    msei = mse/sample
                self.ens_weights[client_idx][m_idx] = 1./(mser + msei + \
                                                          FedAvgEnsAggregatorAuePc.EPS)
            # The most recent model gets the "perfect" score
            self.ens_weights[client_idx][0] = 1./(mser + FedAvgEnsAggregatorAuePc.EPS)
            #normalize
            self.ens_weights[client_idx] = self.ens_weights[client_idx]/self.ens_weights[client_idx].sum()

            logging.info('Ensemble weights for Client {}==>'.format(client_idx))
            logging.info(self.ens_weights[client_idx])        
        
        return

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
                if self.args.is_mobile == 1 and num_sample > 0:
                    model = transform_list_to_tensor(model)
                if num_sample > 0:
                    model_list.append((num_sample, model))
                    training_num += num_sample

            #logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

            # logging.info("################aggregate: %d" % len(model_list))
            (num0, averaged_params) = model_list[0]
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    # Skip the client that doesn't have data for this model
                    if local_sample_number == 0:
                        continue
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # update the global model which is cached at the server side
            self.models[m_idx].load_state_dict(averaged_params)

        # Update ensemble weights
        if round_idx % 10 == 0 or round_idx > (self.args.comm_round - 10):
            self.update_ens_weights()
            
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
        return None

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
                train_tot_correct, train_num_sample, train_loss = self._infer(self.train_data_local_dicts[0][client_idx])
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # test data
                test_tot_correct, test_num_sample = self._infer_ens(
                    self.test_data_local_dicts[0][client_idx],
                    self.ens_weights[client_idx])
                test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                test_num_samples.append(copy.deepcopy(test_num_sample))

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
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            stats = {'test_acc': test_acc}
            logging.info(stats)

    def reported_acc(self, correct, num_sample):
        if num_sample == 0:
            return -1
        else:
            return correct/num_sample

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
        self.models[0].eval()
        self.models[0].to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.models[0](x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        self.models[0].to(torch.device('cpu'))

        return test_acc, test_total, test_loss

    def _infer_ens(self, test_data, ens_weights):
        for model in self.models:
            model.eval()
            model.to(self.device)

        test_acc = test_total = 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                pred_pr = np.zeros((target.size(0), self.class_num))
                x = x.to(self.device)
                target = target.to(self.device)
                # Go through all models
                for model, weight in zip(self.models, ens_weights):
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
        
        for model in self.models:
            model.to(torch.device('cpu'))

        return test_acc, test_total
