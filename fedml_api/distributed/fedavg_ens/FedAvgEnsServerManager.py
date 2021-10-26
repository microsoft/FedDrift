import logging
import torch

from fedml_api.distributed.fedavg_ens.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import transform_tensor_to_list
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class FedAvgEnsServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        # extra info
        extra_info = self.aggregator.extra_info(self.round_idx)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id-1],
                                          extra_info)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        weights_and_num_samples = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_AND_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, weights_and_num_samples)
        
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate(self.round_idx)
            self.aggregator.test_on_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.save_model_params(global_model_params)
                self.finish()
                return

            # sampling clients
            client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                             self.args.client_num_per_round)
            # extra info
            extra_info = self.aggregator.extra_info(self.round_idx)
            
            print("size = %d" % self.size)
            #if self.args.is_mobile == 1:
            #    print("transform_tensor_to_list")
            #    global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id, global_model_params, client_indexes[receiver_id-1], extra_info)

    def send_message_init_config(self, receive_id, global_model_params, client_index, extra_info):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_EXTRA_INFO, extra_info)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, extra_info):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_EXTRA_INFO, extra_info)
        self.send_message(message)
        
    def save_model_params(self, global_model_params):
        param_dict = dict(enumerate(global_model_params))
        torch.save(param_dict, 'model_params.pt')
