import torch
import torchvision

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
    
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
