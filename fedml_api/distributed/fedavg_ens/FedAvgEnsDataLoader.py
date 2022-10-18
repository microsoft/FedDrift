import numpy as np
import pickle
import torch
import json
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import logging
import wandb
from torch import nn
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from fedml_api.model.utils import reinitialize
import random
import copy
import torch.utils.data as todata


# Loader function for AUE
def AUE_data_loader(args, loader_func, device):
    # Determine the number of models in the ensemble
    model_num = min(args.curr_train_iteration + 1, args.ensemble_window)

    datasets = []
    for m in range(model_num):
        args.retrain_data = 'win-{}'.format(m+1)
        datasets.append(loader_func(args))
    
    return datasets
    

class KueState:
    def __init__(self, model_num, feature_num):
        self.model_num = model_num
        self.feature_num = feature_num
        self.worst_idx = 0
        self.masks = np.zeros((model_num, feature_num), dtype=bool)
        
        for model_idx in range(self.model_num):
            self.initialize_mask(model_idx)
    
    def set_worst_idx(self, model_idx):
        self.worst_idx = model_idx
        
    def get_worst_idx(self):
        return self.worst_idx
    
    def get_masks(self):
        return self.masks
        
    def initialize_mask(self, model_idx):
        r = np.random.randint(low=1, high=self.feature_num+1)
        features_used = np.random.choice(self.feature_num, size=r, replace=False)
        for f_idx in features_used:
            self.masks[model_idx][f_idx] = True
        

def Kue_data_loader(args, loader_func, device, comm, process_id):    
    args.retrain_data = 'poisson'
    datasets = []
    for m in range(args.concept_num):    
        datasets.append(loader_func(args))
        
    comm.Barrier()
    if args.curr_train_iteration == 0 and process_id == 0:
        feature_num = datasets[0][-1]
        kue_state = KueState(args.concept_num, feature_num)
        with open('kue_state.pkl','wb') as f:
            pickle.dump(kue_state, f)
    comm.Barrier()

    return datasets
    
    
class AdaState:
    def __init__(self, init_lr=1e-2, beta1=0.5, beta2=0.5, beta3=0.5):
        self.init_lr = init_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        
        self.eta = init_lr
        self.mu = None
        self.s = 0
        self.gam = 0
        
    def update(self, theta, t):
        # count starting at 1 instead of 0
        t = t + 1
        
        # copy estimates from t-1
        prev_mu = self.mu
        if prev_mu is None:
            prev_mu = np.zeros(theta.shape)
        prev_s = self.s
        prev_gam = self.gam        
        if t != 1:
            prev_muh = prev_mu / (1-self.beta1**(t-1))
            prev_sh = prev_s / (1-self.beta2**(t-1))
        else:
            prev_muh = 0
            prev_sh = 0
        
        # compute new estimates for t
        new_mu = self.beta1 * prev_mu + (1-self.beta1) * theta
        new_s = self.beta2 * prev_s + (1-self.beta2) * np.mean((theta-prev_muh)*(theta-prev_muh))
        new_sh = new_s / (1-self.beta2**t)
        
        if prev_sh != 0:
            ratio = new_sh / prev_sh
        else:
            ratio = 1

        new_gam = self.beta3 * prev_gam + (1-self.beta3) * ratio
        new_gamh = new_gam / (1-self.beta3**t)
        
        # store estimates
        self.eta = min(self.init_lr, (self.init_lr * new_gamh)/t)
        
        self.mu = copy.deepcopy(new_mu)
        self.s = new_s
        self.gam = new_gam
        
    def current_lr(self):
        return self.eta


def Ada_data_loader(args, loader_func, device, comm, process_id):
    # Initialize the AdaState
    comm.Barrier()
    if args.curr_train_iteration == 0 and process_id == 0:
        ada_state = AdaState(init_lr=args.lr)
        with open('ada_state.pkl','wb') as f:
            pickle.dump(ada_state, f)
    comm.Barrier()
    
    # the extra algo arg is of the form "{'win-1' OR 'all'}_{'round' OR 'iter'}"
    args.retrain_data = args.concept_drift_algo_arg.split('_')[0]
    
    datasets = []
    datasets.append(loader_func(args))

    return datasets


class DriftSurfState:
    def __init__(self, delta=0.1, r=3, wl=10):
        self.reac_len = r
        self.delta = delta
        self.win_len = wl #Number of batches to consider
        self.models = {'pred':None,'stab':None,'reac':None}
        self.train_data_dict = {'pred':[0],'stab':[0],'reac':None}
        self.train_keys = ['pred','stab']
        self.acc_best = 0
        self.acc_dict = None
        self.reac_ctr = None
        self.state = 'stab'
        self.model_key = 'pred' #Model used for prediction

    def _score(self, model_key, test_data, device):
        self.models[model_key].eval()
        self.models[model_key].to(device)
        test_acc = test_total = 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = self.models[model_key](x)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_total += target.size(0)

        self.models[model_key].to(torch.device('cpu'))
        if test_total == 0:
            return 0
        return test_acc/test_total

    def _append_train_data(self, model_key, iter_id):
        self.train_data_dict[model_key].append(iter_id)
        # Limit the number of data batches based on window length
        if len(self.train_data_dict[model_key]) > self.win_len:
            self.train_data_dict[model_key].pop(0)

    def _reset(self, key):
        self.models[key] = None
        self.train_data_dict[key] = []     

    def move_model_to_cpu(self):
        for key in self.models.keys():
            if self.models[key] is not None:
                self.models[key].cpu()

    def get_train_keys(self):
        return self.train_keys

    def get_train_data(self, key):
        return self.train_data_dict[key]

    def get_model_key(self):
        return self.model_key

    def set_model(self, key, model):
        self.models[key] = model

    def print_model(self):
        for key in self.models.keys():
            if self.models[key] is not None:
                print('Model {}:'.format(key))
                print(self.models[key].state_dict())

    def run_ds_algo(self, new_data, device, curr_iter):        
        acc_pred = self._score('pred', new_data, device)
        print('DS Iteration {}, acc: {}'.format(curr_iter, acc_pred))
        if acc_pred > self.acc_best:
            self.acc_best = acc_pred
        if self.state == 'stab':
            if len(self.train_data_dict['stab']) == 0:
                acc_stab = 0
            else:
                acc_stab = self._score('stab', new_data, device)
            if (acc_pred < self.acc_best - self.delta) or \
               (acc_pred < acc_stab - self.delta/2):
                # Enter reactive state
                self.state = 'reac'
                self._reset('reac')
                self.reac_ctr = 0
                self.acc_dict = {'pred':np.zeros(self.reac_len),
                                 'reac':np.zeros(self.reac_len)}
            else:
                # Stay in stable state
                self._append_train_data('pred', curr_iter)
                self._append_train_data('stab', curr_iter)
                self.train_keys = ['pred', 'stab']
        if self.state == 'reac':
            if self.reac_ctr > 0:
                acc_reac = self._score('reac', new_data, device)
                print('acc_reac = {}'.format(acc_reac))
                self.acc_dict['pred'][self.reac_ctr-1] = acc_pred
                self.acc_dict['reac'][self.reac_ctr-1] = acc_reac
                # Set key for next time step
                if acc_reac > acc_pred:
                    self.model_key = 'reac'
                else:
                    self.model_key = 'pred'
            self._append_train_data('pred', curr_iter)
            self._append_train_data('reac', curr_iter)
            self.train_keys = ['pred', 'reac']
            self.reac_ctr += 1
            if self.reac_ctr == self.reac_len:
                # Exit Reactive State
                self.state = 'stab'
                self._reset('stab')
                if np.mean(self.acc_dict['pred']) < np.mean(self.acc_dict['reac']):
                    self.models['pred'] = self.models['reac']
                    self.train_data_dict['pred'] = self.train_data_dict['reac']
                    self.acc_best = np.amax(self.acc_dict['reac'])
                    self.model_key = 'pred'
                self.acc_dict = None
                self.reac_ctr = None
        # Debug
        print(self.state)
        print(self.train_data_dict)
        print(self.train_keys)
        print(self.acc_best)
        print(self.model_key)
        
# Loader function for DriftSurf
def DriftSurf_data_loader(args, loader_func, device, comm, process_id):
    
    datasets = []
    if args.curr_train_iteration == 0:
        # Hardcoded delta
        deltas = {'sea': 0.02, 'sine': 0.10, 'circle': 0.05}
        
        input_delta = 0.01*float(args.concept_drift_algo_arg)
        if input_delta == 0:
            input_delta = deltas[args.dataset]
        
        ds_state = DriftSurfState(delta=input_delta)
        
        for i in range(2):
            args.retrain_data = 'sel-0'
            datasets.append(loader_func(args))
    else:
        # Load the previous state and models
        with open('ds_state.pkl', 'rb') as f:
            ds_state = pickle.load(f)
        
        # Load the most recent batch of training data
        args.retrain_data = 'win-1'
        data_batch = loader_func(args)
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict,
         test_data_local_dict, class_num, feature_num] = data_batch
         
        ds_state.run_ds_algo(train_data_global, device,
                             args.curr_train_iteration)
        for key in ds_state.get_train_keys():
            train_data = ds_state.get_train_data(key)
            args.retrain_data = 'sel-{}'.format(
                ','.join([str(x) for x in train_data]))
            print(args.retrain_data)
            datasets.append(loader_func(args))
            
    # Save state
    comm.Barrier()
    if process_id == 0:
        ds_state.move_model_to_cpu()
        with open('ds_state.pkl','wb') as f:
            pickle.dump(ds_state, f)
    comm.Barrier()
            
    return datasets


class MultiModelAccState:
    def __init__(self, client_num, model_num=2, delta=0.1):
        self.client_num = client_num
        self.model_num = model_num
        self.delta = delta
        self.train_data_dict = dict()
        for m in range(model_num):
            # each item in the train_data_dict represents one model
            # and its corresponding the training data (by client)
            self.train_data_dict[m] = [[] for c in range(client_num)]            
        self.models = dict()
        self.train_model_idx = dict()
        self.test_model_idx = dict()
        self.acc_dict = dict()        

    def _score(self, model_key, test_data, device):
        self.models[model_key].eval()
        self.models[model_key].to(device)
        test_acc = test_total = 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = self.models[model_key](x)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_total += target.size(0)
        self.models[model_key].to(torch.device('cpu'))
        if test_total == 0:
            return 0
        return test_acc/test_total

    def run_model_select(self, new_data_local_dict, device, curr_iter):
        # Special case for iteration 0, every client contributes to 
        # the first model
        if curr_iter == 0:
            for c in range(self.client_num):
                self.train_data_dict[0][c].append(0)
                self.train_model_idx[c] = 0
                self.test_model_idx[c] = 0
            return

        # Check if all model slots are taken
        next_free_model = -1
        for m in range(self.model_num):
            if m not in self.models:
                next_free_model = m
                break

        # Make the model selection decision for each client
        for c in range(self.client_num):
            model_acc = dict()
            for m in self.models.keys():
                model_acc[m] = self._score(m, new_data_local_dict[c], device)
            best_model = -1
            best_acc = 0.
            for m in model_acc.keys():
                if model_acc[m] > best_acc:
                    best_acc = model_acc[m]
                    best_model = m
            # If the best accuracy drops more than delta, contribute to a new
            # model if it's still possible
            if self.acc_dict[c] - best_acc > self.delta and \
               next_free_model != -1:
                best_model = next_free_model

            self.train_data_dict[best_model][c].append(curr_iter)
            self.train_model_idx[c] = best_model
            self.test_model_idx[c] = best_model

        # DEBUG
        print('train data dict ==>')
        print(self.train_data_dict)

    def model_select_geni(self, curr_iter, change_points, time_stretch):
        # Only works for two models and one change point
        for c in range(self.client_num):
            best_model = change_points[curr_iter//time_stretch][c]
            self.train_data_dict[best_model][c].append(curr_iter)
            self.train_model_idx[c] = best_model
            self.test_model_idx[c] = best_model

    def model_select_geniex(self, curr_iter, change_points, time_stretch):
        # Only works for two models and one change point
        # This one can predict which model for testing based on
        # the oracle knowledge
        min_cp = 1000000
        for t in range(change_points.shape[0]):
            if any(change_points[t]):
                min_cp = t * time_stretch
                break
                
        for c in range(self.client_num):
            train_model = change_points[curr_iter//time_stretch][c]
            if curr_iter >= min_cp:
                test_model = change_points[(curr_iter+1)//time_stretch][c]
            else:
                test_model = train_model
                
            self.train_data_dict[train_model][c].append(curr_iter)
            self.train_model_idx[c] = train_model
            self.test_model_idx[c] = test_model

    def set_model(self, key, model):
        self.models[key] = model

    def set_acc(self, client, acc):
        self.acc_dict[client] = acc

    def move_model_to_cpu(self):
        for key in self.models.keys():
            if self.models[key] is not None:
                self.models[key].cpu()

    def get_train_data_by_model(self, key):
        # Make sure there is data for this model
        train_data = self.train_data_dict[key]
        has_data = False
        for dl in train_data:
            if len(dl) > 0:
                has_data = True
                break
        if not has_data:
            return ''

        return json.dumps(train_data)

    def get_test_model_idx(self, client_idx):
        return self.test_model_idx[client_idx]

    def get_train_model_idx(self, client_idx):
        return self.train_model_idx[client_idx]
            
                
def MultiModelAcc_data_loader(args, loader_func, device, comm, process_id):
    datasets = []
    # Hardcoded delta
    deltas = {'sea': 0.04, 'sine': 0.20, 'circle': 0.10, 'MNIST': 0.10}
    
    if args.curr_train_iteration == 0:
        mm_state = MultiModelAccState(args.client_num_in_total,
                                      args.concept_num,
                                      deltas[args.dataset])
        mm_state.run_model_select(None, device,
                                  args.curr_train_iteration)
    else:
        # Load the previous state and models
        with open('mm_state.pkl', 'rb') as f:
            mm_state = pickle.load(f)
        
        # Load the most recent batch of training data
        args.retrain_data = 'win-1'
        data_batch = loader_func(args)
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict,
         test_data_local_dict, class_num, feature_num] = data_batch
        mm_state.run_model_select(train_data_local_dict, device,
                                  args.curr_train_iteration)

    # Load data by model
    for m in range(args.concept_num):
        train_data = mm_state.get_train_data_by_model(m)
        if train_data != '':
            print('Model {} training data = {}'.format(m, train_data))
            args.retrain_data = 'clientsel-' + train_data
            datasets.append(loader_func(args))

    # Save state
    comm.Barrier()
    if process_id == 0:
        mm_state.move_model_to_cpu()
        with open('mm_state.pkl','wb') as f:
            pickle.dump(mm_state, f)
    comm.Barrier()

    return datasets


def MultiModelGeni_data_loader(args, loader_func, device, comm, process_id):
    datasets = []
    change_points = np.loadtxt("./../../../data/changepoints/{0}.cp".format(args.change_points), dtype=np.dtype(int))
            
    if args.curr_train_iteration == 0:
        mm_state = MultiModelAccState(args.client_num_in_total,
                                      args.concept_num)
    else:
        # Load the previous state and models
        with open('mm_state.pkl', 'rb') as f:
            mm_state = pickle.load(f)        
    
    mm_state.model_select_geni(args.curr_train_iteration,
                               change_points,
                               args.time_stretch)

    # Load data by model
    for m in range(args.concept_num):
        train_data = mm_state.get_train_data_by_model(m)
        if train_data != '':
            print('Model {} training data = {}'.format(m, train_data))
            args.retrain_data = 'clientsel-' + train_data
            datasets.append(loader_func(args))

    # Save state
    comm.Barrier()
    if process_id == 0:
        mm_state.move_model_to_cpu()
        with open('mm_state.pkl','wb') as f:
            pickle.dump(mm_state, f)
    comm.Barrier()

    return datasets


def MultiModelGeniEx_data_loader(args, loader_func, device, comm, process_id):
    datasets = []
    change_points = np.loadtxt("./../../../data/changepoints/{0}.cp".format(args.change_points), dtype=np.dtype(int))
            
    if args.curr_train_iteration == 0:
        mm_state = MultiModelAccState(args.client_num_in_total,
                                      args.concept_num)
    else:
        # Load the previous state and models
        with open('mm_state.pkl', 'rb') as f:
            mm_state = pickle.load(f)        
    
    mm_state.model_select_geniex(args.curr_train_iteration,
                                 change_points,
                                 args.time_stretch)

    # Load data by model
    for m in range(args.concept_num):
        train_data = mm_state.get_train_data_by_model(m)
        if train_data != '':
            print('Model {} training data = {}'.format(m, train_data))
            args.retrain_data = 'clientsel-' + train_data
            datasets.append(loader_func(args))

    # Save state
    comm.Barrier()
    if process_id == 0:
        mm_state.move_model_to_cpu()
        with open('mm_state.pkl','wb') as f:
            pickle.dump(mm_state, f)
    comm.Barrier()

    return datasets

def ClusterFL_data_loader(args, loader_func, device, comm, process_id):
    datasets = []

    model_num = args.concept_num
    for m in range(model_num):
        args.retrain_data = args.concept_drift_algo_arg
        datasets.append(loader_func(args))
    
    return datasets

# Note: this data loader is currently created for Lin and Exp 
# for which the Trainers ignore these data and instead use all_local_data
def SingleModel_data_loader(args, loader_func, device, comm, process_id):
    args.retrain_data = 'win-1'
    return [loader_func(args)]
    
class SoftClusterState:
    def __init__(self, client_num, model_num=2, cluster_alg='softmax_0', 
                 mmacc_delta=0.1, softmax_alpha=0, geni_change_points=None, geni_stretch=1,
                 h_delta=0.1, h_deltap=0.1, h_w=1, h_distance='A', h_cluster='C',
                 cfl_gamma=0.1, cfl_retrain='win-1'):
        self.client_num = client_num
        self.model_num = model_num
        # keys are iterations, and values are weight matrices (model_num x client_num)
        self.train_data_weights = dict()
        
        self.cluster_alg = cluster_alg
        self.mmacc_delta = mmacc_delta
        self.mmacc_acc_dict = dict()
        self.softmax_alpha = softmax_alpha
        self.geni_change_points = geni_change_points
        self.geni_stretch = geni_stretch
        
        self.h_delta = h_delta
        self.h_deltap = h_deltap
        self.h_w = h_w
        self.h_distance = h_distance
        self.h_cluster = h_cluster
        
        # keys are clients, and values are pairs (model_idx, time to unmark)
        self.h_marked = {}
        self.h_next_free_model = 1
        
        self.cfl_gamma = cfl_gamma
        self.cfl_retrain = cfl_retrain
        self.cfl_norm = 0   # the max observed mean norm
        self.cfl_eps1 = 0
        self.cfl_eps2 = 10000
        
        
    # Every client contributes to the first model
    def cluster_init(self):
        self.train_data_weights[0] = np.zeros((self.model_num, self.client_num))
        
        if self.h_cluster == 'F':
            for c in range(self.client_num):
                self.train_data_weights[0][c][c] = 1.
            for c in range(self.client_num):
                wandb.log({"Plurality/CL-{}".format(c): c, "round": 0})
                wandb.run.summary["Contribute/CL-{}".format(c)] = 1
            wandb.run.summary["num_models"] = self.client_num   
            wandb.run.summary["local_models"] = self.client_num
            return
        
        for c in range(self.client_num):
            self.train_data_weights[0][0][c] = 1.
            
        for c in range(self.client_num):
            wandb.log({"Plurality/CL-{}".format(c): 0, "round": 0})
            
            wandb.run.summary["Contribute/CL-{}".format(c)] = 1
        
        wandb.run.summary["num_models"] = 1   
        wandb.run.summary["local_models"] = 0
        
    def cluster(self, acc_matrix, curr_iter, round_idx):
        if self.cluster_alg == "hard":
            self.cluster_hard(acc_matrix, curr_iter)
        elif self.cluster_alg == "hard-r":
            self.cluster_hard(acc_matrix, curr_iter)
        elif 'softmax' in self.cluster_alg:
            self.cluster_softmax(acc_matrix, curr_iter)
        elif 'mmacc' in self.cluster_alg:
            if round_idx == 0:
                self.cluster_mmacc(acc_matrix, curr_iter)
            else:
                self.cluster_hard_among_existing(acc_matrix, curr_iter)
        elif self.cluster_alg == 'gmm':
            self.cluster_gmm(acc_matrix, curr_iter)
        elif self.cluster_alg == 'geni':
            if round_idx == 0:
                self.cluster_geni(curr_iter)           
        else:
            raise NameError('cluster alg')
            
        for c in range(self.client_num):
            # if self.model_num == 2:   # handled this case separately if weights are fractional
                # wandb.log({"Weight-1/CL-{}".format(c): self.train_data_weights[curr_iter][1][c],
                           # "round": round_idx})
            # else:
            wandb.log({"Plurality/CL-{}".format(c): self.get_test_model_idx(curr_iter, c), 
                       "round": round_idx})
            if 'softmax' in self.cluster_alg:
                wandb.log({"Weight-All/CL-{}".format(c): np.array2string(self.train_data_weights[curr_iter][:,c]),
                           "round": round_idx})
            
    def cluster_hard(self, acc_matrix, curr_iter):
        # initialize weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        # list of model_idx with best acc by client
        best_models = np.argmax(acc_matrix, axis=0)
        # assign weight 1 to the best model at each client
        for c in range(self.client_num):
            self.train_data_weights[curr_iter][best_models[c]][c] = 1.
    
    def cluster_softmax(self, acc_matrix, curr_iter):
        # softmax over accuracies per client
        self.train_data_weights[curr_iter] = softmax(acc_matrix*(2**self.softmax_alpha), axis=0)
    
    # replicate mmacc. only makes sense to run this once per iter
    def cluster_mmacc(self, acc_matrix, curr_iter):
        # clustering occurs only among models already initialized and not reset
        models_in_use = []
        for m in range(self.model_num):
            if any( any( self.train_data_weights[t][m] > 0 ) for t in range(curr_iter) ):
                models_in_use.append(m)
                
        # initialize weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        
        # first identify with best model among models_in_use 
        for c in range(self.client_num):
            best_model_idx = np.argmax(acc_matrix[models_in_use,c])
            best_model = models_in_use[best_model_idx]
            self.train_data_weights[curr_iter][best_model][c] = 1.
        
        next_free_model = -42   # special initial value until some client detects a drift
        
        # switch to a new model if acc degrades
        for c in range(self.client_num):
            best_model_idx = np.argmax(acc_matrix[models_in_use,c])
            best_model = models_in_use[best_model_idx]
            
            newest_acc = acc_matrix[best_model][c]
            
            if self.mmacc_acc_dict[c] - acc_matrix[best_model][c] > self.mmacc_delta:
                if next_free_model == -42:
                    next_free_model = self.find_unused_model_lru(curr_iter)             
                if next_free_model != -1:
                    for mmm in range(self.model_num):
                        self.train_data_weights[curr_iter][mmm][c] = 0
                    self.train_data_weights[curr_iter][next_free_model][c] = 1.
            
            self.set_acc(c, newest_acc)
            
        self.log_models(curr_iter)
            
    def log_models(self, curr_iter):
        num_models = 0
                
        if self.h_cluster == 'E':
            # for feddrift-c, count models prior to current time step, and add 1 if any drift for next time
            for m in range(self.model_num):
                if any( any( self.train_data_weights[t][m] > 0 ) for t in range(curr_iter) ):
                    num_models += 1
            if len(self.h_marked) > 0:
                num_models += 1      
        else:
            for m in range(self.model_num):
                if any( any( self.train_data_weights[t][m] > 0 ) for t in range(curr_iter+1) ):
                    num_models += 1
                
        wandb.run.summary["num_models"] = num_models
        
        # set of clients that train a given model        
        trained_by = {}     
        for m in range(self.model_num):
            trained_by[m] = set()
        for t in range(curr_iter+1):
            for m in range(self.model_num):
                for c in range(self.client_num):
                    if self.train_data_weights[t][m][c] > 0:
                        trained_by[m].add(c)
                        
        # filter out models that are trained by a single client
        local_models = 0
        for m in range(self.model_num):
            if len(trained_by[m]) == 1:
                local_models += 1
                del trained_by[m]
        wandb.run.summary["local_models"] = local_models
        
        for c in range(self.client_num):
            num_trained = 0
            for m, clients in trained_by.items():
                if c in clients:
                    num_trained += 1
            
            wandb.run.summary["Contribute/CL-{}".format(c)] = num_trained
        
    def cluster_hard_among_existing(self, acc_matrix, curr_iter):
        # clustering occurs only among models currently in use
        models_in_use = []
        for m in range(self.model_num):
            if any( self.train_data_weights[curr_iter][m] > 0 ):
                models_in_use.append(m)
        
        # reset the weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        # identify best model among models_in_use
        for c in range(self.client_num):
            best_model_idx = np.argmax(acc_matrix[models_in_use,c])
            best_model = models_in_use[best_model_idx]               
            self.train_data_weights[curr_iter][best_model][c] = 1.

    def cluster_gmm(self, acc_matrix, curr_iter):
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))

        gm = GaussianMixture(n_components=2, random_state=0).fit(acc_matrix.transpose())
        probs = gm.predict_proba(acc_matrix.transpose()).transpose()
        
        # heuristic for mapping cluster to model id
        if gm.means_[0][0] > gm.means_[0][1]:
            self.train_data_weights[curr_iter][0] = probs[0]
            self.train_data_weights[curr_iter][1] = probs[1]
        else:
            self.train_data_weights[curr_iter][0] = probs[1]
            self.train_data_weights[curr_iter][1] = probs[0]
            
    def cluster_mmacc2(self, curr_iter, models, all_data, device):
        acc_matrix = self.train_acc_matrix(curr_iter, models, all_data, device, list(range(self.model_num)))
        
        # clustering occurs only among models already initialized and not reset
        models_in_use = []
        for m in range(self.model_num):
            if any( any( self.train_data_weights[t][m] > 0 ) for t in range(curr_iter) ):
                models_in_use.append(m)
                
        # initialize weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        # first identify with best model among models_in_use 
        for c in range(self.client_num):
            best_model_idx = np.argmax(acc_matrix[models_in_use,c])
            best_model = models_in_use[best_model_idx]
            self.train_data_weights[curr_iter][best_model][c] = 1.
        
        next_free_model = -42   # special initial value until some client detects a drift
        
        # switch to a new model if acc degrades
        for c in range(self.client_num):
            best_model_idx = np.argmax(acc_matrix[models_in_use,c])
            best_model = models_in_use[best_model_idx]
            
            newest_acc = acc_matrix[best_model][c]
            
            if self.mmacc_acc_dict[c] - acc_matrix[best_model][c] > self.mmacc_delta:
                if next_free_model == -42:
                    next_free_model = self.find_unused_model_lru(curr_iter, models, best_model)             
                if next_free_model != -1:
                    for mmm in range(self.model_num):
                        self.train_data_weights[curr_iter][mmm][c] = 0
                    self.train_data_weights[curr_iter][next_free_model][c] = 1.
            
            self.set_acc(c, newest_acc)
            
        # log clustering
        for c in range(self.client_num):
            wandb.log({"Plurality/CL-{}".format(c): self.get_test_model_idx(curr_iter, c), "round": 0})
            
        self.log_models(curr_iter)
        
            
    def cluster_hierarchical(self, curr_iter, models, all_data, device):
        # FedDrift-C: delete all but one model created
        if self.h_cluster == 'E':
            marked_models = [m for (m, t) in self.h_marked.values()]
            if len(marked_models) != 0:
                model_keep = np.random.choice(marked_models)
                for mm in marked_models:
                    if mm != model_keep:
                        reinitialize(models[mm])
                        self.set_weights_zero_model(mm)      
        
        # models can leave isolation
        self.update_marking(curr_iter)
        
        # clustering occurs only among models already initialized and not marked
        models_in_use = []
        marked_models = [m for (m, t) in self.h_marked.values()]
        for m in range(self.model_num):
            if any( any( self.train_data_weights[t][m] > 0 ) for t in range(curr_iter) ) and m not in marked_models:
                models_in_use.append(m)
                
        # get accuracy of each model in use at each client at this iter
        acc_matrix = self.train_acc_matrix(curr_iter, models, all_data, device, models_in_use)
                
        # initialize the weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        # marked clients stay on their local model
        for c, (m, t) in self.h_marked.items():
            self.train_data_weights[curr_iter][m][c] = 1.
            
        # first set everybody to their best existing model (for the purpose of not evicting under LRU)
        for c in range(self.client_num):
            if c not in self.h_marked:
                best_model_idx = np.argmax(acc_matrix[:,c])
                best_model = models_in_use[best_model_idx]
                self.train_data_weights[curr_iter][best_model][c] = 1.
        
        # unmarked clients choose among models_in_use or they leave if there's a drift
        for c in range(self.client_num):
            if c not in self.h_marked:
                best_model_idx = np.argmax(acc_matrix[:,c])
                best_model = models_in_use[best_model_idx]
                
                newest_acc = acc_matrix[best_model_idx][c]
                
                if self.mmacc_acc_dict[c] - acc_matrix[best_model_idx][c] > self.h_delta:
                    next_free_model = self.find_unused_model_lru(curr_iter, models, best_model)
                    if next_free_model != -1:
                        best_model = next_free_model
                        self.h_marked[c] = (best_model, curr_iter + self.h_w)
                        
                        for mmm in range(self.model_num):
                            self.train_data_weights[curr_iter][mmm][c] = 0
                        
                        self.train_data_weights[curr_iter][best_model][c] = 1.
                
                self.set_acc(c, newest_acc)
        
        if len(models_in_use) > 1:
            # identify global data for each model
            cluster_data = {}
            # case: data is list of batches
            if all( isinstance(all_data[c][0], list) for c in range(self.client_num) ):
                for m in models_in_use:
                    cluster_data[m] = []
                    for c in range(self.client_num):
                        for t in range(curr_iter+1):
                            if self.train_data_weights[t][m][c] == 1:
                                cluster_data[m] += all_data[c][t]
                    random.shuffle(cluster_data[m])
            # case: data is torch dataloader
            else:
                for m in models_in_use:
                    m_datasets = []
                    for c in range(self.client_num):
                        for t in range(curr_iter+1):
                            if self.train_data_weights[t][m][c] == 1:
                                if len(all_data[c][t]) > 0:
                                    m_datasets.append(all_data[c][t].dataset)
                    cluster_data[m] = todata.DataLoader(todata.ConcatDataset(m_datasets),
                                                        batch_size=32, shuffle=True)
                                         
            # compute cluster accuracy matrix
            cluster_acc = np.zeros((len(models_in_use),len(models_in_use)))
            for i in range(len(models_in_use)):
                model = models[models_in_use[i]]
                for j in range(len(models_in_use)):
                    data = cluster_data[models_in_use[j]]
                    num_correct, num_sample, _ = self._infer_subset(model, data, device, 20)
                    if num_sample != 0:
                        cluster_acc[i][j] = num_correct/num_sample

            # compute distance matrix
            distance_matrix = np.zeros((len(models_in_use),len(models_in_use)))
            for i in range(len(models_in_use)):
                for j in range(len(models_in_use)):
                    if self.h_distance == 'A':
                        distance_matrix[i][j] = max(cluster_acc[i][i] - cluster_acc[i][j],
                                                    cluster_acc[j][j] - cluster_acc[j][i],
                                                    0)
                    elif self.h_distance == 'B':
                        distance_matrix[i][j] = max(cluster_acc[i][i] - cluster_acc[j][i],
                                                    cluster_acc[j][j] - cluster_acc[i][j],
                                                    0)
            
            # hierarchically cluster until delta'
            if self.h_cluster == 'C' or self.h_cluster == 'E' or self.h_cluster == 'F':
                Z = sch.linkage(squareform(distance_matrix), method='complete')
            elif self.h_cluster == 'D':
                Z = sch.linkage(squareform(distance_matrix), method='average')
            T = sch.fcluster(Z, t=self.h_deltap, criterion='distance')
            
            clusters = {}
            for i in range(len(models_in_use)):
                cluster_id = T[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(models_in_use[i])
                
            # log merging
            list_log = []
            for to_merge in clusters.values():
                if len(to_merge) > 1:
                    list_log.append('(' + ', '.join(str(e) for e in to_merge) + ')')
            if len(list_log) > 0:
                wandb.run.summary["Merge"] = ', '.join(list_log)
                
            for to_merge in clusters.values():
                base_model = to_merge[0]
                for j in range(1, len(to_merge)):
                    second_model = to_merge[j]
                    self.merge(curr_iter, models, base_model, second_model)
                
        # log clustering
        for c in range(self.client_num):
            wandb.log({"Plurality/CL-{}".format(c): self.get_test_model_idx(curr_iter, c), "round": 0})
            
        self.log_models(curr_iter)
            
    
    # policy: give up on allocating a new model if exceeded the cap
    def find_unused_model_capped(self):
        next_free_model = -1
        if self.h_next_free_model < self.model_num:
            next_free_model = self.h_next_free_model
            self.h_next_free_model += 1
        return next_free_model
    
    # policy: look for a currently unused model after exceeding the cap
    def find_unused_model(self, curr_iter, models):
        if self.h_next_free_model < self.model_num:
            next_free_model = self.h_next_free_model
            self.h_next_free_model += 1
        else:
            next_free_model = 0
            for i in range(self.model_num):
                m = (i + self.h_next_free_model) % self.model_num                
                if not (any( self.train_data_weights[curr_iter-1][m] ) 
                     or any( self.train_data_weights[curr_iter][m]   )):
                    next_free_model = m
            
            self.h_next_free_model = self.model_num + (next_free_model + 1)
            self.set_weights_zero_model(next_free_model)
            reinitialize(models[next_free_model])
            
        return next_free_model
        
    # LRU policy after exceeding the cap
    # returns -1 if all models are currently used 
    # TODO: may be possible to do something smarter for feddrift-c
    def find_unused_model_lru(self, curr_iter, models = [], original_model = 0):
        if self.h_next_free_model < self.model_num:
            next_free_model = self.h_next_free_model
            self.h_next_free_model += 1
        else:
            time_last_used = -1 * np.ones(self.model_num)
            
            for t in range(curr_iter+1):
                for m in range(self.model_num):
                    if any( self.train_data_weights[t][m] ):
                        time_last_used[m] = t
            
            lru_models = np.where(time_last_used == time_last_used.min())[0]
            
            next_free_model = np.random.choice(lru_models)
            
            if time_last_used[next_free_model] == curr_iter:
                return -1
            
            self.set_weights_zero_model(next_free_model)
        if len(models) != 0:
            # option for initializing with previous parameters
            models[next_free_model].load_state_dict(models[original_model].state_dict())
            # reinitialize(models[next_free_model])
            
        return next_free_model
    
    def update_marking(self, curr_iter):
        to_unmark = []
        
        for c, (m, t) in self.h_marked.items():
            if t == curr_iter:
                to_unmark.append(c)
        
        for c in to_unmark:
            del self.h_marked[c]
            
    def merge(self, curr_iter, models, base_model, second_model):
        w1 = 0
        w2 = 0
        for c in range(self.client_num):
            for t in range(curr_iter+1):
                w1 += self.train_data_weights[t][base_model][c]
                w2 += self.train_data_weights[t][second_model][c]
        s = w1 + w2
        w1 = w1/s
        w2 = w2/s
        
        model1 = models[base_model].state_dict()
        model2 = models[second_model].state_dict()
        
        for k in model1.keys():
            model1[k] = model1[k] * w1 + model2[k] * w2
        
        models[base_model].load_state_dict(model1)
        reinitialize(models[second_model])
        
        for c in range(self.client_num):
            for t in range(curr_iter+1):
                self.train_data_weights[t][base_model][c] += self.train_data_weights[t][second_model][c]
    
        self.set_weights_zero_model(second_model)    
    
    def train_acc_matrix(self, curr_iter, models, all_data, device, models_in_use):
        acc = np.zeros((len(models_in_use), self.client_num))
        
        for row in range(len(models_in_use)):
            m = models_in_use[row]
            for c in range(self.client_num):
                data = all_data[c][curr_iter]
                num_correct, num_sample, _ = self._infer(models[m], data, device)
                if num_sample != 0:
                    acc[row][c] = num_correct/num_sample
        
        return acc

    def _infer(self, model, test_data, device):
        model.eval()
        model.to(device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        model.to(torch.device('cpu'))

        return test_acc, test_total, test_loss

    # subset_size indicates max number of batches from test_data to evaluate
    def _infer_subset(self, model, test_data, device, subset_size):
        batch_count = 0
        
        model.eval()
        model.to(device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
                
                batch_count += 1
                if batch_count > subset_size:
                    break

        model.to(torch.device('cpu'))

        return test_acc, test_total, test_loss

    # replicate mmgeni
    def cluster_geni(self, curr_iter):
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        for c in range(self.client_num):
            best_model = self.geni_change_points[curr_iter//self.geni_stretch][c]
            self.train_data_weights[curr_iter][best_model][c] = 1.


            
    def cluster_cfl_init(self, curr_iter):
        self.train_data_weights[curr_iter] = np.copy(self.train_data_weights[curr_iter-1])
        if self.cfl_retrain == 'win-1':
            self.set_weights_win1(curr_iter)
        
        for c in range(self.client_num):
            wandb.log({"Plurality/CL-{}".format(c): self.get_test_model_idx(curr_iter, c), 
                       "round": 0})
            
    def cluster_cfl(self, curr_iter, round_idx, models, weights_dict):
        did_split = False
        
        # identify each active cluster
        models_in_use = []
        for m in range(self.model_num):
            if any( self.train_data_weights[curr_iter][m] > 0 ):
                models_in_use.append(m)
        
        # identify clients at each cluster
        clusters = {}
        for m in models_in_use:
            clusters[m] = np.nonzero(self.train_data_weights[curr_iter][m])[0]
        
        # check for a binary split at each cluster
        for model_idx, clients in clusters.items():
            
            old_model = models[model_idx].cpu().state_dict()
            
            weight_updates = []
            for c in clients:
                local_model, local_samples = weights_dict[c][model_idx]
                if local_samples == 0:
                    continue
                diff = dict()
                for k in old_model.keys():
                    diff[k] = local_model[k] - old_model[k]
                weight_updates.append(diff)
            
            max_norm = self.cfl_util_max_norm(weight_updates)
            mean_norm = self.cfl_util_mean_norm(weight_updates)         
        
            if mean_norm > self.cfl_norm:
                self.cfl_norm = mean_norm
                self.cfl_eps1 = self.cfl_norm/10.0
                self.cfl_eps2 = 6 * self.cfl_eps1
            else:
                if mean_norm < self.cfl_eps1 and max_norm > self.cfl_eps2:
                    similarities = self.cfl_util_pairwise_similarities(weight_updates)
                    cl1, cl2 = self.cfl_util_bipartition(similarities)
                    alpha_cross = max( max( similarities[i, j] for j in cl2) for i in cl1)
                    
                    if ((1-alpha_cross)/2.0)**0.5 > self.cfl_gamma:
                        next_free_model = self.find_unused_model_capped()                    
                        if next_free_model != -1:
                            did_split = True
                            reinitialize(models[model_idx])
                            self.train_data_weights[curr_iter][model_idx] = np.zeros(self.client_num)
                            
                            for i in cl1:
                                client_idx = clients[i]
                                self.train_data_weights[curr_iter][model_idx][client_idx] = 1.
                            for i in cl2:
                                client_idx = clients[i]
                                self.train_data_weights[curr_iter][next_free_model][client_idx] = 1.        
        
        if did_split:              
            for c in range(self.client_num):
                wandb.log({"Plurality/CL-{}".format(c): self.get_test_model_idx(curr_iter, c), 
                           "round": round_idx})
            if self.cfl_retrain == 'all':
                for t in range(curr_iter):
                    self.train_data_weights[t] = np.copy(self.train_data_weights[curr_iter])
            
        return did_split
        
    def cfl_util_flatten(self, source):
        return np.concatenate([value.flatten() for value in source.values()],
                              axis = None)
        
    def cfl_util_max_norm(self, weight_updates):
        return np.max([np.linalg.norm(self.cfl_util_flatten(dW))
                       for dW in weight_updates])
        
    def cfl_util_mean_norm(self, weight_updates):
        return np.linalg.norm(np.mean(np.stack([self.cfl_util_flatten(dW) for dW in weight_updates]), axis=0))
        
    def cfl_util_pairwise_similarities(self, weight_updates):
        angles = np.zeros((len(weight_updates), len(weight_updates)))
        for i, source1 in enumerate(weight_updates):
            for j, source2 in enumerate(weight_updates):
                s1 = self.cfl_util_flatten(source1)
                s2 = self.cfl_util_flatten(source2)
                angles[i,j] = np.dot(s1,s2)/(np.linalg.norm(s1)*np.linalg.norm(s2)+1e-12)
        return angles
        
    def cfl_util_bipartition(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
        cl1 = np.argwhere(clustering.labels_ == 0).flatten()
        cl2 = np.argwhere(clustering.labels_ == 1).flatten()
        return cl1, cl2
        


    # extra state maintained for mmacc clustering
    def set_acc(self, client, acc):
        self.mmacc_acc_dict[client] = acc

    def get_test_model_idx(self, curr_iter, client_idx):
        return np.argmax(self.train_data_weights[curr_iter][:,client_idx])
    
    def get_weights(self):
        return self.train_data_weights
         
    def set_weights_win1(self, curr_iter):
        for t in range(curr_iter):
            self.train_data_weights[t] = np.zeros((self.model_num, self.client_num))
    
    def set_weights_zero_model(self, m_idx):
        for t in self.train_data_weights.keys():
            self.train_data_weights[t][m_idx] = np.zeros(self.client_num)

    
def SoftCluster_data_loader(args, loader_func, device, comm, process_id):
    datasets = []
    default_deltas = {'sea': 0.04, 'sine': 0.20, 'circle': 0.10, 'MNIST': 0.10}
    
    # Initialize the SoftClusterState with input args
    comm.Barrier()
    if args.curr_train_iteration == 0 and process_id == 0:
        cluster_alg = args.concept_drift_algo_arg
        mmacc_delta = 0
        softmax_alpha = 0
        geni_change_points = None
        geni_stretch = 1
        h_delta = 0
        h_deltap = 0
        h_w = 0
        h_distance = ''
        h_cluster = ''
        cfl_gamma = 0
        cfl_retrain = ''
        
        if 'mmacc' in cluster_alg:
            mmacc_delta = 0.01*float(cluster_alg.split('_')[-1])
            if mmacc_delta == 0 and args.dataset in default_deltas:
                mmacc_delta = default_deltas[args.dataset]
        elif 'softmax' in cluster_alg:
            softmax_alpha = int(cluster_alg.split('_')[-1])
        elif cluster_alg == 'geni':
            geni_change_points = np.loadtxt("./../../../data/changepoints/{0}.cp".format(args.change_points), dtype=np.dtype(int))
            geni_stretch = args.time_stretch
        elif 'H' in cluster_alg:    # string is of the form "H_{distance fn}_{cluster distance fn}_{W}_{100*delta}_{100*delta'}"
            h_distance = cluster_alg.split('_')[1]
            h_cluster = cluster_alg.split('_')[2]
            h_w = int(cluster_alg.split('_')[3])
            h_delta = 0.01*float(cluster_alg.split('_')[4])
            if h_delta == 0 and args.dataset in default_deltas:
                h_delta = default_deltas[args.dataset]
            h_deltap = 0.01*float(cluster_alg.split('_')[5])
            if h_deltap == 0:
                h_deltap = h_delta
        elif 'cfl' in cluster_alg:  # string is of the form "cfl_{gamma}_{win-1 OR all}"
            cfl_gamma = float(cluster_alg.split('_')[1])
            cfl_retrain = cluster_alg.split('_')[2]

        sc_state = SoftClusterState(args.client_num_in_total,
                                    args.concept_num,
                                    cluster_alg,
                                    mmacc_delta,
                                    softmax_alpha,
                                    geni_change_points,
                                    geni_stretch,
                                    h_delta,
                                    h_deltap,
                                    h_w,
                                    h_distance,
                                    h_cluster,
                                    cfl_gamma,
                                    cfl_retrain)
                                    
        with open('sc_state.pkl','wb') as f:
            pickle.dump(sc_state, f)
    comm.Barrier()

    # Load data by model. Actually, these training data are ignored by the TrainerSoftCluster
    # which instead uses its all_local_data that is partitioned by iteration
    args.retrain_data = 'win-1'
    data_per_model = loader_func(args)
    for m in range(args.concept_num):
        datasets.append(data_per_model)

    return datasets
