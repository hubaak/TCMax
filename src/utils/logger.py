import os

import torch
from torch.utils.tensorboard import SummaryWriter

class LOGGER():
    def __init__(self, args):
        self.args = args
        self.enable_tb = args.use_tensorboard
        
        # tensorboard
        if self.enable_tb:
            if not os.path.exists(args.tensorboard_path):
                os.mkdir(args.tensorboard_path)
            writer_path = os.path.join(args.tensorboard_path, args.dataset)
            if not os.path.exists(writer_path):
                os.mkdir(writer_path)
            log_name = '{}_{}_{}'.format(args.fusion_method, args.modulation, args.loss_type)
            if args.tensorboard_suffix != '':
                log_name = log_name + '_' + args.tensorboard_suffix
            self.writer = SummaryWriter(os.path.join(writer_path, log_name))
        
        self.statistics_in_epoch = dict()
        self.statistics_counter = {key:0 for key in self.statistics_in_epoch.keys()}
        self.epoch = 0
        self.iter_n = 0
        
        
    def before_epoch(self, epoch, split = 'train'):
        self.epoch = epoch
        for key in self.statistics_in_epoch.keys():
            self.statistics_in_epoch[key] = 0.0
            self.statistics_counter[key] = 0
    
    def after_epoch(self, split = 'train'):
        for key in self.statistics_in_epoch.keys():
            if self.statistics_counter[key] !=0:
                self.statistics_in_epoch[key] /= self.statistics_counter[key]
        if split == 'train':
            if self.enable_tb:
                for key in self.statistics_in_epoch.keys():
                    if self.statistics_counter[key] !=0:
                        self.tb_writer_epoch(key, self.statistics_in_epoch[key])
            print_str = "Loss: {:.3f}".format(self.statistics_in_epoch['Loss/Total Loss'])
            for key in ['Audio Loss', 'Video Loss', 'Text Loss']:
                if 'Loss/' + key in self.statistics_in_epoch.keys():
                    print_str += ", {}: {:.3f}".format(key, self.statistics_in_epoch['Loss/' + key])
            print(print_str)
        elif split == 'val':
            pass

    def before_iter(self, batch_size, split = 'train'):
        pass
    
    def after_iter(self, loss_and_acc:dict, split = 'train'):
        self.iter_n += 1
        for key in loss_and_acc.keys():
            value = loss_and_acc[key]
            if value == None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.statistics_add(key, value)
    
    def tb_writer_epoch(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self.enable_tb:
            self.writer.add_scalar(name, value, self.epoch)  
          
    def tb_writer_iter(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self.enable_tb:
            self.writer.add_scalar(name, value, self.iter_n)
    
    def statistics_add(self, key, value):
        if key not in self.statistics_in_epoch.keys():
            self.statistics_in_epoch[key] = 0.0
            self.statistics_counter[key] = 0
        self.statistics_in_epoch[key] += value
        self.statistics_counter[key] += 1
        