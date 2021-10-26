import numpy as np
import pickle
import torch
import json
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
import logging

# Loader function for AUE
def AUE_data_loader(args, loader_func, device):
    # Determine the number of models in the ensemble
    model_num = min(args.curr_train_iteration + 1, args.ensemble_window)

    datasets = []
    for m in range(model_num):
        args.retrain_data = 'win-{}'.format(m+1)
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
        # TODO: accept argments for DriftSurf parameters
        ds_state = DriftSurfState(delta=deltas[args.dataset])
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
         test_data_local_dict, all_data, class_num, feature_num] = data_batch
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

    def model_select_geni(self, curr_iter, change_points):
        # Only works for two models and one change point
        for c in range(self.client_num):
            cp = change_points[c]
            if curr_iter >= cp:
                best_model = 1
            else:
                best_model = 0            
            self.train_data_dict[best_model][c].append(curr_iter)
            self.train_model_idx[c] = best_model
            self.test_model_idx[c] = best_model

    def model_select_geniex(self, curr_iter, change_points):
        # Only works for two models and one change point
        # This one can predict which model for testing based on
        # the oracle knowledge
        min_cp = 1000000
        for c in range(self.client_num):
            if change_points[c] < min_cp:
                min_cp = change_points[c]
                
        for c in range(self.client_num):
            cp = change_points[c]
            if curr_iter >= cp:
                train_model = 1
            else:
                train_model = 0

            if curr_iter >= (cp-1) and cp > min_cp:
                test_model = 1
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
         test_data_local_dict, all_data, class_num, feature_num] = data_batch
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
    # Load the change points
    data_path = './../../../data/{}/'.format(args.dataset)
    change_points = {}
    with open(data_path + 'change_points', 'r') as cpf:
        for c, line in enumerate(cpf):
            change_points[c] = int(line.strip())
            
    if args.curr_train_iteration == 0:
        mm_state = MultiModelAccState(args.client_num_in_total,
                                      args.concept_num)
    else:
        # Load the previous state and models
        with open('mm_state.pkl', 'rb') as f:
            mm_state = pickle.load(f)        
    
    mm_state.model_select_geni(args.curr_train_iteration,
                               change_points)

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
    # Load the change points
    data_path = './../../../data/{}/'.format(args.dataset)
    change_points = {}
    with open(data_path + 'change_points', 'r') as cpf:
        for c, line in enumerate(cpf):
            change_points[c] = int(line.strip())
            
    if args.curr_train_iteration == 0:
        mm_state = MultiModelAccState(args.client_num_in_total,
                                      args.concept_num)
    else:
        # Load the previous state and models
        with open('mm_state.pkl', 'rb') as f:
            mm_state = pickle.load(f)        
    
    mm_state.model_select_geniex(args.curr_train_iteration,
                                 change_points)

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

    model_num = 2  # Hardcoded 2 models for now
    for m in range(model_num):
        args.retrain_data = args.concept_drift_algo_arg
        datasets.append(loader_func(args))
    
    return datasets
    
class SoftClusterState:
    def __init__(self, client_num, model_num=2, cluster_alg='softmax_0', 
                 mmacc_delta=0.1, softmax_alpha=0, geni_change_points={}):
        self.client_num = client_num
        self.model_num = model_num
        # keys are iterations, and values are weight matrices (model_num x client_num)
        self.train_data_weights = dict()
        
        self.cluster_alg = cluster_alg
        self.mmacc_delta = mmacc_delta
        self.mmacc_acc_dict = dict()
        self.softmax_alpha = softmax_alpha
        self.geni_change_points = geni_change_points
        
    # Every client contributes to the first model
    def cluster_init(self):
        self.train_data_weights[0] = np.zeros((self.model_num, self.client_num))
        for c in range(self.client_num):
            self.train_data_weights[0][0][c] = 1.
        
    def cluster(self, acc_matrix, curr_iter):
        if self.cluster_alg == "hard":
            self.cluster_hard(acc_matrix, curr_iter)
        elif 'softmax' in self.cluster_alg:
            self.cluster_softmax(acc_matrix, curr_iter)
        elif 'mmacc' in self.cluster_alg:
            self.cluster_mmacc(acc_matrix, curr_iter)
        elif self.cluster_alg == 'gmm':
            self.cluster_gmm(acc_matrix, curr_iter)
        elif self.cluster_alg == 'geni':
            self.cluster_geni(curr_iter)
        else:
            raise NameError('cluster alg')
            
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
        # clustering occurs only among models already initialized        
        last_model_in_use = -1
        for m in reversed(range(self.model_num)):
            if any( self.train_data_weights[curr_iter-1][m] > 0 ):
                last_model_in_use = m
                break
        models_used = last_model_in_use + 1
        if models_used == self.model_num:
            next_free_model = -1
        else:
            next_free_model = last_model_in_use + 1
        
        # initialize weight matrix
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        # identify best model among models_used, and switch to a new model if acc degrades
        for c in range(self.client_num):
            best_model = np.argmax(acc_matrix[:models_used,c])
            if self.mmacc_acc_dict[c] - acc_matrix[best_model][c] > self.mmacc_delta and \
               next_free_model != -1:
                best_model = next_free_model                
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

    # replicate mmgeni
    def cluster_geni(self, curr_iter):
        self.train_data_weights[curr_iter] = np.zeros((self.model_num, self.client_num))
        
        for c in range(self.client_num):
            cp = self.geni_change_points[c]
            if curr_iter >= cp:
                best_model = 1
            else:
                best_model = 0
            self.train_data_weights[curr_iter][best_model][c] = 1.

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
            
    def set_weights_zero_b(self):
        for t in self.train_data_weights.keys():
            self.train_data_weights[t][0] = np.ones(self.client_num)
            self.train_data_weights[t][1] = np.zeros(self.client_num)

    
def SoftCluster_data_loader(args, loader_func, device, comm, process_id):
    datasets = []
    default_deltas = {'sea': 0.04, 'sine': 0.20, 'circle': 0.10, 'MNIST': 0.10}
    
    # Initialize the SoftClusterState with input args
    comm.Barrier()
    if args.curr_train_iteration == 0 and process_id == 0:
        cluster_alg = args.concept_drift_algo_arg
        mmacc_delta = 0
        softmax_alpha = 0
        geni_change_points = {}
        
        if 'mmacc' in cluster_alg:
            mmacc_delta = 0.01*float(cluster_alg.split('_')[-1])
            if mmacc_delta == 0 and args.dataset in default_deltas:
                mmacc_delta = default_deltas[args.dataset]
        elif 'softmax' in cluster_alg:
            softmax_alpha = int(cluster_alg.split('_')[-1])
        elif cluster_alg == 'geni':
            data_path = './../../../data/{}/'.format(args.dataset)
            with open(data_path + 'change_points', 'r') as cpf:
                for c, line in enumerate(cpf):
                    geni_change_points[c] = int(line.strip())

        sc_state = SoftClusterState(args.client_num_in_total,
                                    args.concept_num,
                                    cluster_alg,
                                    mmacc_delta,
                                    softmax_alpha,
                                    geni_change_points)
        with open('sc_state.pkl','wb') as f:
            pickle.dump(sc_state, f)
    comm.Barrier()

    # Load data by model. Actually, these training data are ignored by the 
    # TrainerSoftCluster which instead uses its all_local_data
    for m in range(args.concept_num):
        args.retrain_data = 'all'
        datasets.append(loader_func(args))

    return datasets
