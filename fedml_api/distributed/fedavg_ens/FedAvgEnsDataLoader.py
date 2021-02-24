import numpy as np
import pickle
import torch

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
def DriftSurf_data_loader(args, loader_func, device):
    
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
    ds_state.move_model_to_cpu()
    with open('ds_state.pkl','wb') as f:
        pickle.dump(ds_state, f)
            
    return datasets
