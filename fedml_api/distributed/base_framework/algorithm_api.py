from mpi4py import MPI

from fedml_api.distributed.base_framework.central_manager import BaseCentralManager
from fedml_api.distributed.base_framework.central_worker import BaseCentralWorker
from fedml_api.distributed.base_framework.client_manager import BaseClientManager
from fedml_api.distributed.base_framework.client_worker import BaseClientWorker


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_Base_distributed(process_id, worker_number, comm, args):
    if process_id == 0:
        init_central_worker(args, comm, process_id, worker_number)
    else:
        init_client_worker(args, comm, process_id, worker_number)


def init_central_worker(args, comm, process_id, size):
    # aggregator
    client_num = size - 1
    aggregator = BaseCentralWorker(client_num, args)

    # start the distributed training
    server_manager = BaseCentralManager(args, comm, process_id, size, aggregator)
    server_manager.run()


def init_client_worker(args, comm, process_id, size):
    # trainer
    client_ID = process_id - 1
    trainer = BaseClientWorker(client_ID)

    client_manager = BaseClientManager(args, comm, process_id, size, trainer)
    client_manager.run()
