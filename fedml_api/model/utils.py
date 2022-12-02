import torch
import torchvision

# default value, may be reset in main
torch_seed = 42

def reinitialize(model):
    #torch.manual_seed(10)
    
    if isinstance(model, torchvision.models.densenet.DenseNet):
        pretrained_model = torchvision.models.densenet121(pretrained=True)
        model.load_state_dict(pretrained_model.state_dict())
        return
        
    if isinstance(model, torchvision.models.resnet.ResNet):
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        model.load_state_dict(pretrained_model.state_dict())
        return
        
    torch.manual_seed(torch_seed)
    
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
