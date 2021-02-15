from mpi4py import MPI

from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorAue import FedAvgEnsAggregatorAue
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorAuePc import FedAvgEnsAggregatorAuePc
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainer import FedAvgEnsTrainer
from fedml_api.distributed.fedavg_ens.FedAvgEnsClientManager import FedAvgEnsClientManager
from fedml_api.distributed.fedavg_ens.FedAvgEnsServerManager import FedAvgEnsServerManager

from fedml_api.distributed.fedavg_ens.FedAvgEnsDataLoader import AUE_data_loader, DriftSurf_data_loader


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def FedML_FedAvgEns_data_loader(args, loader_func):
    if args.concept_drift_algo in {"aue", "auepc"}:
        return AUE_data_loader(args, loader_func)
    elif args.concept_drift_algo == "driftsurf":
        return DriftSurf_data_loader(args, loader_func)


def FedML_FedAvgEns_distributed(process_id, worker_number, device, comm, models,
                                datasets, class_num, args):
    datasets_t = [list(x) for x in zip(*datasets)]
    [train_data_nums, test_data_nums, train_data_globals, test_data_globals,
     train_data_local_num_dicts, train_data_local_dicts, test_data_local_dicts,
     class_nums, feature_nums] = datasets_t
    
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, models, train_data_nums, train_data_globals,
                    test_data_globals, train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts,
                    class_num)
    else:
        init_client(args, device, comm, process_id, worker_number, models, train_data_nums, train_data_local_num_dicts,
                    train_data_local_dicts)


def init_server(args, device, comm, rank, size, models, train_data_nums, train_data_globals, test_data_globals,
                train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, class_num):
    # aggregator
    worker_num = size - 1
    if args.concept_drift_algo == "aue":
        aggregator = FedAvgEnsAggregatorAue(train_data_globals, test_data_globals, train_data_nums,
                                            train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, worker_num,
                                            device, models, class_num, args)
    elif args.concept_drift_algo == "auepc":
        aggregator = FedAvgEnsAggregatorAuePc(train_data_globals, test_data_globals, train_data_nums,
                                              train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, worker_num,
                                              device, models, class_num, args)
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
def init_client(args, device, comm, process_id, size, model, train_data_nums, train_data_local_num_dicts, train_data_local_dicts):
    # trainer
    client_index = process_id - 1
    trainer = FedAvgEnsTrainer(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, device, models, args)

    client_manager = FedAvgEnsClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
