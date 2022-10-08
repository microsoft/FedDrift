import argparse
import logging
import os
import sys
import torch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.sea.data_loader import generate_data_sea

from fedml_api.data_preprocessing.sine.data_loader import generate_data_sine

from fedml_api.data_preprocessing.circle.data_loader import generate_data_circle

from fedml_api.data_preprocessing.MNIST.data_loader_cont import generate_data_mnist

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')
                        
    parser.add_argument('--sample_num', type=int, default=500,
                        help='number of samples per client per iter')  
                                                    
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')                  

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--train_iteration', type=int, default=3,
                        help='The number of FedML iterations (over time)')

    parser.add_argument('--drift_together', type=int, default=0,
                        help='If the concept drift happens at the same time across all clients')

    parser.add_argument('--change_points', type=str, default='',
                        help='Specify change point matrix (a filename in data dir)')
                        
    parser.add_argument('--time_stretch', type=int, default=1,
                        help='change points are stretched out by this multiplicative factor')
                        
    parser.add_argument('--noise_prob', type=float, default=0,
                        help='label of a sample is swapped with this probability')
    
    args = parser.parse_args()
    return args


def prepare_data(args, dataset_name):
    logging.info("generate_data. dataset_name = %s" % dataset_name)
    if dataset_name == "sea":
        generate_data_sea(args.train_iteration, args.client_num_in_total,
                          args.drift_together, args.sample_num, args.noise_prob, args.time_stretch, args.change_points)

    elif dataset_name == "sine":
        logging.info("generate_data. dataset_name = %s" % dataset_name)
        generate_data_sine(args.train_iteration, args.client_num_in_total,
                           args.drift_together, args.sample_num, args.noise_prob, args.time_stretch, args.change_points)

    elif dataset_name == "circle":
        logging.info("generate_data. dataset_name = %s" % dataset_name)
        generate_data_circle(args.train_iteration, args.client_num_in_total,
                             args.drift_together, args.sample_num, args.noise_prob, args.time_stretch, args.change_points)
    
    elif dataset_name == "MNIST":
        logging.info("generate_data. dataset_name = %s" % dataset_name)
        generate_data_mnist(args.train_iteration, args.client_num_in_total,
                            args.drift_together, args.sample_num, args.noise_prob, args.time_stretch, args.change_points)
        
    return

if __name__ == "__main__":
    logging.basicConfig(filename='data_output.log', level=logging.INFO)

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
    
    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # Prepare data
    prepare_data(args, args.dataset)



