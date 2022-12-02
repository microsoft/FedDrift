import argparse
import logging
import os
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
import pickle
import random

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.sea.data_loader import load_partition_data_sea, load_all_data_sea

from fedml_api.data_preprocessing.sine.data_loader import load_partition_data_sine, load_all_data_sine

from fedml_api.data_preprocessing.circle.data_loader import load_partition_data_circle, load_all_data_circle

from fedml_api.data_preprocessing.MNIST.data_loader_cont import load_partition_data_mnist, load_all_data_mnist

from fedml_api.data_preprocessing.fmow.data_loader import load_partition_data_fmow, load_all_data_fmow

from fedml_api.model.linear.lr import LogisticRegression

from fedml_api.model.fnn.fnn import FeedForwardNN

from fedml_api.model.cv.cnn import CNN_DropOut

import torchvision

from fedml_api.model import utils

from fedml_api.distributed.fedavg_ens.FedAvgEnsAPI import FedML_init, FedML_FedAvgEns_distributed, FedML_FedAvgEns_data_loader


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')                       

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--total_train_iteration', type=int, default=3,
                        help='The number of FedML iterations (over time)')

    parser.add_argument('--curr_train_iteration', type=int, default=0,
                        help='The current Fededrated Learning iterations (over time)')

    parser.add_argument('--drift_together', type=int, default=0,
                        help='If the concept drift happens at the same time across all clients')

    parser.add_argument('--report_client', type=int, default=0,
                        help='Whether reporting the accuracy of each client')

    parser.add_argument('--retrain_data', type=str, default='win-1',
                        help='which data to be included for retraining')

    parser.add_argument('--concept_drift_algo', type=str, default='aue',
                        help='The algorithm to handle concept drift')

    parser.add_argument('--concept_drift_algo_arg', type=str, default='',
                        help='The parameter for concept drift algorithm')

    parser.add_argument('--ensemble_window', type=int, default=4,
                        help='The number of models to keep in the ensemble')

    parser.add_argument('--concept_num', type=int, default=2,
                        help='The number of concepts in the experiments')

    parser.add_argument('--change_points', type=str, default='',
                        help='Specify change point matrix (a filename in data dir)')

    parser.add_argument('--time_stretch', type=int, default=1,
                        help='change points are stretched out by this multiplicative factor')                        
                        
    parser.add_argument('--reset_models', type=int, default=0,
                        help='If the model parameters should be reset between train iterations')
    
    # this parameter is unused after prepare_data but included here to log on wandb
    parser.add_argument('--noise_prob', type=float, default=0,
                        help='label of a sample is swapped with this probability')
                        
    parser.add_argument('--dummy_arg', type=int, default=0,
                        help='parameter to distinguish different trials. may be used for random seeds')

    args = parser.parse_args()
    return args

def load_data_by_dataset(args):
    dataset_name = args.dataset
    logging.info("load_data. dataset_name = %s" % dataset_name)

    if dataset_name == "sea":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_sea(args.batch_size, args.curr_train_iteration,
                                            args.client_num_in_total, args.retrain_data)
        feature_num = 3

    elif dataset_name == "sine":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_sine(args.batch_size, args.curr_train_iteration,
                                             args.client_num_in_total, args.retrain_data)
        feature_num = 2

    elif dataset_name == "circle":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_circle(args.batch_size, args.curr_train_iteration,
                                               args.client_num_in_total, args.retrain_data)
        feature_num = 2
        
    elif dataset_name == "MNIST":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size, args.curr_train_iteration,
                                              args.client_num_in_total, args.retrain_data)
        feature_num = 784
        
    elif dataset_name == "fmow":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_fmow(args.batch_size, args.curr_train_iteration,
                                             args.client_num_in_total, args.retrain_data, 
                                             args.data_dir, args.change_points)
        feature_num = 3*224*224

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
               class_num, feature_num]
    return dataset
    

def load_all_data_by_dataset(args):
    dataset_name = args.dataset
    
    if dataset_name == "sea":
        return load_all_data_sea(args.batch_size, args.curr_train_iteration, args.client_num_in_total)

    elif dataset_name == "sine":
        return load_all_data_sine(args.batch_size, args.curr_train_iteration, args.client_num_in_total)

    elif dataset_name == "circle":
        return load_all_data_circle(args.batch_size, args.curr_train_iteration, args.client_num_in_total)
        
    elif dataset_name == "MNIST":
        return load_all_data_mnist(args.batch_size, args.curr_train_iteration, args.client_num_in_total)
        
    elif dataset_name == "fmow":
        return load_all_data_fmow(args.batch_size, args.curr_train_iteration, args.client_num_in_total, 
                                  args.data_dir, args.change_points)


def create_model(args, model_name, output_dim, feature_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr":
        logging.info("LogisticRegression, feature_dim = %s" % feature_dim)
        model = LogisticRegression(feature_dim, output_dim)
    if model_name == "fnn":
        logging.info("FeedForwardNN, feature_dim = %s" % feature_dim)
        model = FeedForwardNN(feature_dim, output_dim, feature_dim * 2)
    if model_name == "cnn":
        logging.info("CNN_DropOut")
        model = CNN_DropOut()
    if model_name == "densenet":
        model = torchvision.models.densenet121(pretrained=True)
    if model_name == "resnet":
        model = torchvision.models.resnet18(pretrained=True)
    utils.reinitialize(model)
    return model

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    logging.basicConfig(filename='output.log', level=logging.INFO)

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
#    args.curr_train_iteration -= 1

    # this shouldn't be necessary, but explictly delete state files from
    # previous runs so that initialization bugs will be found at runtime 
    comm.Barrier()
    if args.curr_train_iteration == 0 and process_id == 0:
        filenames = ['model_params.pt', 'ds_state.pkl', 'mm_state.pkl', 'sc_state.pkl', 'ada_state.pkl', 'kue_state.pkl']
        for f in filenames:
            if os.path.exists(f):
                os.remove(f)
    comm.Barrier()

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project="fedml",
            name="FedAvgCont(d)-" + args.dataset +
                "-r" + str(args.comm_round) + "-e" +
                str(args.epochs) + "-lr" + str(args.lr) +
                "-iter" + str(args.curr_train_iteration) +
                "-dt" + str(args.drift_together) +
                "-" + args.concept_drift_algo,
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # Fixed so that we can reproduce the result.
    np.random.seed(args.dummy_arg)
    torch.manual_seed(args.dummy_arg)
    random.seed(args.dummy_arg)
    utils.torch_seed = args.dummy_arg

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    # load data
    datasets = FedML_FedAvgEns_data_loader(args, load_data_by_dataset,
                                           device, comm, process_id)
    all_data = load_all_data_by_dataset(args)
                                           
    #dataset = load_data(args)
    #[train_data_num, test_data_num, train_data_global, test_data_global,
    # train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
    # class_num, feature_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    models = []
    class_num = datasets[0][-2]
    feature_num = datasets[0][-1]
    for m in range(len(datasets)):
        models.append(create_model(args, model_name=args.model, output_dim=class_num,
                                   feature_dim = feature_num))

    # load params among the prev existing models
    if args.curr_train_iteration != 0 and not args.reset_models:
        model_params = torch.load('model_params.pt')
        
        # special case for mm-variants and 2 concepts:
        # after all clients finish transition A->B, only load params of model B
        if args.concept_drift_algo in {"mmacc", "mmgeni", "mmgeniex"} and \
            len(models) == 1 and len(model_params) == 2:
                models[0].load_state_dict(model_params[1])
        
        # AUE: circular
        elif args.concept_drift_algo in {"aue", "auepc"}:
            for m_idx in range(1, len(models)):
                models[m_idx].load_state_dict(model_params[m_idx-1])
        
        # for driftsurf, handled separately by the aggregator
        elif args.concept_drift_algo == "driftsurf":
            pass
        
        elif args.concept_drift_algo == "clusterfl":
            # this is obsolete. instead should use the softcluster algo with the cfl arg
            pass
        
        # general case: load models in same order they were saved
        else:
            for m_idx, p in model_params.items():
                models[m_idx].load_state_dict(p)

    # start "federated averaging (FedAvg) with ensembled" for this round
    FedML_FedAvgEns_distributed(process_id, worker_number, device, comm,
                                models, datasets, all_data, class_num, args)
