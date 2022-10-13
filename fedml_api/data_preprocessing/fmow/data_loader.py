import numpy as np
import torch
import torchvision
import torch.utils.data as todata
from wilds import get_dataset
from scipy.stats import poisson

def load_all_data_fmow(batch_size, current_train_iteration, num_client, 
                       data_dir, partition_name):
    return [ [ create_dataloader(FmowDataset(c, it, data_dir, partition_name), batch_size)
               for it in range(current_train_iteration+1) ]
               for c in range(num_client) ]
    

def load_partition_data_fmow(batch_size, current_train_iteration, num_client, 
                             retrain_data, data_dir, partition_name):
    # Prepare data for FedML
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    # the global data are left uninitialized. in current implementation, no one uses these
    win1_global_dataset = todata.ConcatDataset([ FmowDataset(c, current_train_iteration, data_dir, partition_name) for c in range(num_client) ])
    train_data_global = create_dataloader(win1_global_dataset, batch_size)
    test_data_global = None

    for c in range(num_client):
        iters = retrain_iters(retrain_data, c, current_train_iteration)
        if len(iters) > 1:
            train_data = todata.ConcatDataset([ FmowDataset(c, it, data_dir, partition_name) for it in iters ])
        elif len(iters) == 1:
            train_data = FmowDataset(c, iters[0], data_dir, partition_name)
            if retrain_data.startswith("poisson") and len(train_data) > 1:
                weights = poisson.rvs(mu=1, size=len(train_data))
                if sum(weights) != 0:
                    train_data.subidxs = np.random.choice(train_data.subidxs, size=len(train_data), replace=True, p=weights/sum(weights))
        else:
            train_data = []
        test_data = FmowDataset(c, current_train_iteration+1, data_dir, partition_name)
        
        train_data_num += len(train_data)
        test_data_num += len(test_data)
        train_data_local_num_dict[c] = len(train_data)

        train_data_local_dict[c] = create_dataloader(train_data, batch_size)
        test_data_local_dict[c] = create_dataloader(test_data, batch_size) 
            
    client_num = num_client
    class_num = 1000 # only 62 in the data, but the pre-trained resnet/densenet will output 1000

    return client_num, train_data_num, test_data_num, train_data_global, \
        test_data_global, train_data_local_num_dict, train_data_local_dict, \
        test_data_local_dict, class_num
    

def create_dataloader(dataset, batch_size):
    if len(dataset) == 0:
        return []
    return todata.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    

class FmowDataset(todata.Dataset):
    def __init__(self, client_idx, iteration, 
                 data_dir="./../../../data/fmow/data", partition_name="A"):
        sub_filename = "./../../../data/fmow/partitions/{0}/client_{1}_iter_{2}.csv".format(partition_name, client_idx, iteration)
        self.subidxs = np.loadtxt(sub_filename, dtype=int, delimiter=',')
        # special case for numpy arrays
        if self.subidxs.size == 1:
            self.subidxs = [self.subidxs.item()]
        self.dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=False)
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.subidxs)
        
    def __getitem__(self, idx):
        i = self.subidxs[idx]
        (x_image, y_tensor, metadata) = self.dataset[i]
        x_tensor = self.transform(x_image)
        return (x_tensor, y_tensor)


# analogous to common.retrain.load_retrain_table_data
def retrain_iters(retrain_method, client_idx, current_train_iteration):
    if retrain_method == "all":
        return list(range(current_train_iteration + 1))
    elif retrain_method.startswith("win-"):
        win_size = int(retrain_method.replace("win-", ""))
        start_iter = max(0, current_train_iteration - win_size + 1)
        return list(range(start_iter, current_train_iteration + 1))
    elif retrain_method.startswith("sel-"):
        select_iter = retrain_method.replace("sel-", "").split(",")
        return [int(it) for it in select_iter]
    elif retrain_method.startswith("clientsel-"):
        client_select_iter = json.loads(retrain_method.replace("clientsel-", ""))
        return client_select_iter[client_idx]
    elif retrain_method.startswith("poisson"):
        return [current_train_iteration]
    else:
        raise NameError(retrain_method)
