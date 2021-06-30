import numpy as np
import pickle
import torch
import json

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
    deltas = {'sea': 0.04, 'sine': 0.20, 'circle': 0.10}
    
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
