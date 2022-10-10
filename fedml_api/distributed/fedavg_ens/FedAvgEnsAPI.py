from mpi4py import MPI

from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorAue import FedAvgEnsAggregatorAue
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorAuePc import FedAvgEnsAggregatorAuePc
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorDriftSurf import FedAvgEnsAggregatorDriftSurf
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorMultiModelAcc import FedAvgEnsAggregatorMultiModelAcc
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorClusterFL import FedAvgEnsAggregatorClusterFL
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorSoftCluster import FedAvgEnsAggregatorSoftCluster
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorAda import FedAvgEnsAggregatorAda
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorVanilla import FedAvgEnsAggregatorVanilla
from fedml_api.distributed.fedavg_ens.FedAvgEnsAggregatorKue import FedAvgEnsAggregatorKue
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainer import FedAvgEnsTrainer
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerClusterFL import FedAvgEnsTrainerClusterFL
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerSoftCluster import FedAvgEnsTrainerSoftCluster
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerAda import FedAvgEnsTrainerAda
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerExp import FedAvgEnsTrainerExp
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerLin import FedAvgEnsTrainerLin
from fedml_api.distributed.fedavg_ens.FedAvgEnsTrainerKue import FedAvgEnsTrainerKue
from fedml_api.distributed.fedavg_ens.FedAvgEnsClientManager import FedAvgEnsClientManager
from fedml_api.distributed.fedavg_ens.FedAvgEnsServerManager import FedAvgEnsServerManager

from fedml_api.distributed.fedavg_ens.FedAvgEnsDataLoader import AUE_data_loader, DriftSurf_data_loader, MultiModelAcc_data_loader, MultiModelGeni_data_loader, MultiModelGeniEx_data_loader, ClusterFL_data_loader, SoftCluster_data_loader, Ada_data_loader, SingleModel_data_loader, Kue_data_loader


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def FedML_FedAvgEns_data_loader(args, loader_func, device, comm, process_id):
    if args.concept_drift_algo in {"aue", "auepc"}:
        return AUE_data_loader(args, loader_func, device)
    elif args.concept_drift_algo == "driftsurf":
        return DriftSurf_data_loader(args, loader_func, device,
                                     comm, process_id)
    elif args.concept_drift_algo == "mmacc":
        return MultiModelAcc_data_loader(args, loader_func, device,
                                         comm, process_id)
    elif args.concept_drift_algo == "mmgeni":
        return MultiModelGeni_data_loader(args, loader_func, device,
                                          comm, process_id)
    elif args.concept_drift_algo == "mmgeniex":
        return MultiModelGeniEx_data_loader(args, loader_func, device,
                                            comm, process_id)
    elif args.concept_drift_algo == "clusterfl":
        return ClusterFL_data_loader(args, loader_func, device,
                                     comm, process_id)
    elif args.concept_drift_algo in {"softcluster", "softclusterwin-1", "softclusterreset"}:
        return SoftCluster_data_loader(args, loader_func, device,
                                       comm, process_id)
    elif args.concept_drift_algo == "ada":
        return Ada_data_loader(args, loader_func, device,
                               comm, process_id)
    elif args.concept_drift_algo in {"exp", "lin"}:
        return SingleModel_data_loader(args, loader_func, device,
                                       comm, process_id)
    elif args.concept_drift_algo == "kue":
        return Kue_data_loader(args, loader_func, device,
                               comm, process_id)


def FedML_FedAvgEns_distributed(process_id, worker_number, device, comm, models,
                                datasets, all_data, class_num, args):                    
    train_data_nums = []
    test_data_nums = []
    train_data_globals = []
    test_data_globals = []
    train_data_local_num_dicts = []
    train_data_local_dicts = []
    test_data_local_dicts = []
    for ds in datasets:
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
         class_num, feature_num] = ds
        train_data_nums.append(train_data_num)
        test_data_nums.append(test_data_num)
        train_data_globals.append(train_data_globals)
        test_data_globals.append(test_data_globals)
        train_data_local_num_dicts.append(train_data_local_num_dict)
        train_data_local_dicts.append(train_data_local_dict)
        test_data_local_dicts.append(test_data_local_dict)

    print(train_data_nums)
            
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, models, train_data_nums, train_data_globals,
                    test_data_globals, train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts,
                    all_data, class_num)
    else:
        init_client(args, device, comm, process_id, worker_number, models, train_data_nums, train_data_local_num_dicts,
                    train_data_local_dicts, all_data[process_id - 1])


def init_server(args, device, comm, rank, size, models, train_data_nums, train_data_globals, test_data_globals,
                train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, class_num):
    # aggregator
    worker_num = size - 1
    if args.concept_drift_algo == "aue":
        aggregator = FedAvgEnsAggregatorAue(train_data_globals, test_data_globals, train_data_nums,
                                            train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                            device, models, class_num, args)
    elif args.concept_drift_algo == "auepc":
        aggregator = FedAvgEnsAggregatorAuePc(train_data_globals, test_data_globals, train_data_nums,
                                              train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                              device, models, class_num, args)
    elif args.concept_drift_algo == "driftsurf":
        aggregator = FedAvgEnsAggregatorDriftSurf(train_data_globals, test_data_globals, train_data_nums,
                                                  train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                                  device, models, class_num, args)
    elif args.concept_drift_algo in {"mmacc", "mmgeni", "mmgeniex"}:
        aggregator = FedAvgEnsAggregatorMultiModelAcc(train_data_globals, test_data_globals, train_data_nums,
                                                      train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                                      device, models, class_num, args)
    elif args.concept_drift_algo == "clusterfl":
        aggregator = FedAvgEnsAggregatorClusterFL(train_data_globals, test_data_globals, train_data_nums,
                                                  train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                                  device, models, class_num, args)
    elif args.concept_drift_algo in {"softcluster", "softclusterwin-1", "softclusterreset"}:
        aggregator = FedAvgEnsAggregatorSoftCluster(train_data_globals, test_data_globals, train_data_nums,
                                                    train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                                    device, models, class_num, args)
    elif args.concept_drift_algo == "ada":
        aggregator = FedAvgEnsAggregatorAda(train_data_globals, test_data_globals, train_data_nums,
                                            train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                            device, models, class_num, args)
    elif args.concept_drift_algo in {"exp", "lin"}:
        aggregator = FedAvgEnsAggregatorVanilla(train_data_globals, test_data_globals, train_data_nums,
                                                train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
                                                device, models, class_num, args)
    elif args.concept_drift_algo == "kue":
        aggregator = FedAvgEnsAggregatorKue(train_data_globals, test_data_globals, train_data_nums,
                                            train_data_local_dicts, test_data_local_dicts, train_data_local_num_dicts, all_data, worker_num,
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
def init_client(args, device, comm, process_id, size, models, train_data_nums, train_data_local_num_dicts, train_data_local_dicts, all_local_data):
    # trainer
    client_index = process_id - 1
    if args.concept_drift_algo == "clusterfl":
        trainer = FedAvgEnsTrainerClusterFL(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                            device, models, args)
    elif args.concept_drift_algo in {"softcluster", "softclusterwin-1", "softclusterreset"}:
        trainer = FedAvgEnsTrainerSoftCluster(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                   device, models, args)
    elif args.concept_drift_algo == "ada":
        trainer = FedAvgEnsTrainerAda(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                      device, models, args)
    elif args.concept_drift_algo == "exp":
        trainer = FedAvgEnsTrainerExp(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                      device, models, args)
    elif args.concept_drift_algo == "lin":
        trainer = FedAvgEnsTrainerLin(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                      device, models, args)
    elif args.concept_drift_algo == "kue":
        trainer = FedAvgEnsTrainerKue(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                      device, models, args)
    else:
        trainer = FedAvgEnsTrainer(client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data,
                                   device, models, args)

    client_manager = FedAvgEnsClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
