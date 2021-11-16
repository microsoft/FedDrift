import torch

def reinitialize(model):
    # torch.manual_seed(10)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
