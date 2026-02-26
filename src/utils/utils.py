import torch
import torch.nn as nn
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def args_mute_output(args):
    args.__setattr__('use_tqdm', False)
    args.__setattr__('use_tensorboard', False)
    args.__setattr__('save_model', False)
    return args

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.replace('module.', '') in model.module.pretrained_names:
            param.requires_grad = False
        # if "fusion_module" not in name:  
        #     param.requires_grad = False
        # else:
        #     param.requires_grad = True
            
def unfreeze_backbone(model):
    for name, param in model.named_parameters():
        if name.replace('module.', '') in model.module.pretrained_names:
            param.requires_grad = True