from mpi4py import MPI

from fedml_api.distributed.fedavg.FedAvgEnsAggregatorAue import FedAvgEnsAggregatorAue
from fedml_api.distributed.fedavg.FedAvgEnsTrainer import FedAvgEnsTrainer
from fedml_api.distributed.fedavg.FedAvgEnsClientManager import FedAvgEnsClientManager
from fedml_api.distributed.fedavg.FedAvgEnsServerManager import FedAvgEnsServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvgEns_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, prev_models, class_num, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    prev_models, class_num)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, prev_models, class_num):
    # aggregator
    worker_num = size - 1
    if args.concept_drift_algo == "aue":
        aggregator = FedAvgEnsAggregatorAue(train_data_global, test_data_global, train_data_num,
                                            train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num,
                                            device, model, prev_models, class_num, args)
    else:
        raise NameError('concept_drift_algo')

    # start the distributed training
    server_manager = FedAvgEnsServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()

# TODO List   
# - client should be modified to decentralized worker
# - add group id 
# - Add MPC related setting
def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict, train_data_local_dict):
    # trainer
    client_index = process_id - 1
    trainer = FedAvgEnsTrainer(client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model, args)

    client_manager = FedAvgEnsClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
