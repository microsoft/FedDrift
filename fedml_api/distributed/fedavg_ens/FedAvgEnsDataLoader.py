import numpy as np

# Loader function for AUE
def AUE_data_loader(args, loader_func):
    # Determine the number of models in the ensemble
    model_num = min(args.curr_train_iteration + 1, args.ensemble_window)

    datasets = []
    for m in range(model_num):
        args.retrain_data = 'win-{}'.format(m+1)
        datasets.append(loader_func(args))
    
    return datasets

# Loader function for DriftSurf
def DriftSurf_data_loader(args, loader_func):
    return
