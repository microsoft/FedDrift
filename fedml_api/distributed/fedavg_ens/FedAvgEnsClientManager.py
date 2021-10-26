import logging

from fedml_api.distributed.fedavg_ens.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class FedAvgEnsClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        extra_info = msg_params.get(MyMessage.MSG_ARG_KEY_EXTRA_INFO)

        self.trainer.update_model(global_model_params, extra_info)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        extra_info = msg_params.get(MyMessage.MSG_ARG_KEY_EXTRA_INFO)

        self.trainer.update_model(model_params, extra_info)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds:
            self.finish()

    def send_model_to_server(self, receive_id, weights_and_num_samples):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_AND_NUM_SAMPLES, weights_and_num_samples)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights_and_num_samples = self.trainer.train()
        self.send_model_to_server(0, weights_and_num_samples)
