import argparse
import os

import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CREMADataset, KineticsSound, AVEDataset, VGGSound, UCF101Dataset
from src.models.basic_model import AVClassifier, VTClassifier, VVClassifier, Embed_Classifier

from src.utils.utils import setup_seed, weight_init, args_mute_output, create_dir, freeze_backbone, unfreeze_backbone
from src.utils.losses import *
from src.utils.modulation import *
from src.utils.logger import LOGGER

from src.models.basic_model import get_classes_num

AV_Datasets = ['CREMAD', 'KineticsSound', 'AVE', 'VGGSound']
VV_Datasets = ['UCF101']
Embed_Datasets = [] # For clip

def print_trainable_weight(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def get_arguments():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        choices = AV_Datasets + VT_Datasets + AVT_Datasets + VV_Datasets + Embed_Datasets)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--frames', default=3, type=int)
    parser.add_argument('--audio_path', default=None, type=str)
    parser.add_argument('--visual_path', default=None, type=str)
    parser.add_argument('--text_path', default=None, type=str)
    parser.add_argument('--test_audio_path', default=None, type=str)
    parser.add_argument('--test_visual_path', default=None, type=str)
    parser.add_argument('--test_text_path', default=None, type=str)
    
    # methods
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'MLA', 'AGM', 'MMPareto'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'CLIP', 'concat', 'gated', 'bigated', 'film', 'share_head', 'CLIP_Style', 'concat_msh'])
    parser.add_argument('--loss_type', default='CE', type=str,
                        choices=['CE', 'TCMax', 'Unimodal', 'PMR', 'QMF', 'OPM'])
    
    # batch size and epoch
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    
    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'AdamW'])
    parser.add_argument('--scheduler', default='StepLR', type=str, choices=['StepLR', 'CosineLR', 'ReduceLROnPlateau'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--learning_rate_audio', default=None, type=float, help='initial learning rate')
    parser.add_argument('--learning_rate_visual', default=None, type=float, help='initial learning rate')
    parser.add_argument('--learning_rate_text', default=None, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--weight_decay_audio', default=None, type=float, help='weight decay')
    parser.add_argument('--weight_decay_visual', default=None, type=float, help='weight decay')
    parser.add_argument('--weight_decay_text', default=None, type=float, help='weight decay')
    parser.add_argument('--warm_up_epochs', default=0, type=int, help='')
    
    # backbones
    parser.add_argument('--audio_backbone', default='RN18', type=str, choices=['RN18'])
    parser.add_argument('--visual_backbone', default='RN18', type=str, choices=['RN18', 'ViT_B_32', 'ViT_B_16', 'RN18_pretrained', 'CLIP_RN50', 'CLIP_ViT-B/32', 'CLIP_ViT-B/16', 'RN34_pretrained', 'RN50_pretrained'])
    parser.add_argument('--text_backbone', default='DistilBert', type=str, choices=['Bert', 'DistilBert', 'CLIP_RN50', 'CLIP_ViT-B/32', 'CLIP_ViT-B/16', 'transformer'])
    parser.add_argument('--OF_frames', type=int, default=3)
    
    # Task-Trained weight
    parser.add_argument('--audio_weight', default=None, type=str, help='path to audio weight')
    parser.add_argument('--visual_weight', default=None, type=str, help='path to visual weight')
    
    
    # modulation parameters 
    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    
    # OGM_GE parameters 
    parser.add_argument('--OGM_alpha', default=0.8, type=float, help='alpha in OGM-GE')
    # OPM parameters
    parser.add_argument('--OPM_qbase', default=0.5*0.5, type=float, help='alpha in OGM-GE')
    parser.add_argument('--OPM_lambda', default=1.0, type=float, help='alpha in OGM-GE')
    
    # PMR parameters
    parser.add_argument('--PMR_alpha', default=1.0, type=float, help='alpha in OGM-GE(control the degree of modulation)')
    parser.add_argument('--PMR_epsilon', default=0, type=float, help='epsilon in OGM-GE(prototype update momentum)')
    parser.add_argument('--PMR_mu', default=0.0, type=float, help='mu in OGM-GE(PER weight)')
    
    # AGM parameters
    parser.add_argument('--AGM_alpha',type=float,default=1.0,help='alpha in AGM(degree of Gradient Modulation)')
    
    # TCMax parameters
    parser.add_argument('--TCMax_sample_num', default=None, type=int)
    
    # train or test
    parser.add_argument('--ckpt_path', default=None, type=str, help='path to save trained models')
    parser.add_argument('--task', default='train', type=str, choices=['train', 'val'])
    
    # display and tensorboard
    parser.add_argument('--use_tqdm', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='./output', type=str, help='path to save tensorboard logs')
    parser.add_argument('--tensorboard_suffix', default='', type=str, help='tensorboard log name suffix')
    
    # general settings
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--mute_save_middle_model', action='store_true')
    
    # Mutliple rounds settings
    parser.add_argument('--rounds', default=1, type=int)
    
    # Experiment settings
    parser.add_argument('--save_feature_path', default=None, type=str, help='where to save feature of modality-specific encoders, set None if dont want to save feature')
    
    return parser.parse_args()

# for dual modalities dataset
def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, logger):
    criterion = nn.CrossEntropyLoss()
    n_classes = get_classes_num(args)
    len_dataloader = len(dataloader)
    model.train()
    
    logger.before_epoch(epoch)
    
    if args.use_tqdm:
        data_loader = tqdm(dataloader, file=sys.stdout)
    else:
        data_loader = dataloader
    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()
        if args.dataset in AV_Datasets:
            spec, image, label, idx = batch
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
        elif args.dataset in VT_Datasets:
            image = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            idx = batch['idx']
        elif args.dataset in VV_Datasets:
            image1 = batch['images1'].to(device)
            image2 = batch['images2'].to(device)
            label = batch['label'].to(device)
            idx = batch['idx']
        elif args.dataset in Embed_Datasets:
            embed1 = batch['Embed1'].to(device)
            embed2 = batch['Embed2'].to(device)
            label = batch['label'].to(device)
            idx = batch['idx']
        else:
            raise NotImplementedError(f"{args.datase} dataset is not support")

        logger.before_iter(label.shape[0])
        loss_a = loss_v = loss_t = None
        if args.dataset in AV_Datasets + VV_Datasets + Embed_Datasets:
            if args.dataset in AV_Datasets:
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            elif args.dataset in Embed_Datasets:
                a, v, out = model(embed1.float(), embed2.float())
            else:
                a, v, out = model(image1.float(), image2.float())
                
            if args.modulation == 'MLA':
                loss, loss_a, loss_v = MLA_modulation(args, model, criterion, optimizer, a, v, label, step, len_dataloader)
            elif args.modulation == 'MMPareto':
                if args.fusion_method != 'concat_msh':
                    raise NotImplementedError('MMPareto only support concat_msh fusion!')
                out_a = model.module.fusion_module.forward_a(a)
                out_v = model.module.fusion_module.forward_v(v)
                loss, loss_a, loss_v = MMPareto(args, model, out, out_a, out_v, label, optimizer)
            else:
                out_a = model.module.fusion_module.forward_a(a)
                out_v = model.module.fusion_module.forward_v(v)
                
                if args.loss_type == 'CE':
                    loss = criterion(out, label)
                    loss_a = criterion(out_a, label)
                    loss_v = criterion(out_v, label)
                elif args.loss_type == 'TCMax':
                    if args.TCMax_sample_num is None:
                        loss, loss_a, loss_v = TCMax_loss(args, model, a, v, label)
                    else:
                        loss, loss_a, loss_v = TCMax_loss_sample(args, model, a, v, label, n_classes)
                elif args.loss_type == 'PMR':
                    loss, loss_a, loss_v = model.module.PMR_module.PMR_loss(out, a, v, label, criterion, device, logger)
                elif args.loss_type == 'QMF':
                    loss, loss_a, loss_v = QMF_loss(model, out_a, out_v, label, idx)
                elif args.loss_type == 'OPM':
                    loss, loss_a, loss_v = OPM_loss(args, model, a, v, out_a, out_v, label, device)
                elif args.loss_type == 'Unimodal':
                    loss, loss_a, loss_v = Unimodal_loss(args, out_a, out_v, label, criterion)
                else:
                    raise NotImplementedError
                loss.backward()
        elif args.dataset in VT_Datasets:
            v, t, out = model(image.float(), (input_ids, attention_mask))
            if args.modulation == 'MLA':
                loss, loss_v, loss_t = MLA_modulation(args, model, criterion, optimizer, v, t, label, step, len_dataloader)
            else:
                out_v = model.module.fusion_module.forward_a(v)
                out_t = model.module.fusion_module.forward_v(t)
                
                if args.loss_type == 'CE':
                    loss = criterion(out, label)
                    loss_v = criterion(out_v, label)
                    loss_t = criterion(out_t, label)
                elif args.loss_type == 'TCMax':
                    loss, loss_v, loss_t = TCMax_loss(args, model, v, t, label)
                    loss = loss_v + loss_t
                elif args.loss_type == 'PMR':
                    loss, loss_v, loss_t = model.module.PMR_module.PMR_loss(out, v, t, label, criterion, device, logger)
                elif args.loss_type == 'QMF':
                    loss, loss_v, loss_t = QMF_loss(model, out_v, out_t, label, idx)
                elif args.loss_type == 'Unimodal':
                    loss, loss_v, loss_t = Unimodal_loss(args, out_v, out_t, label, criterion)
                elif args.loss_type == 'JLwInfoNCE':
                    loss = criterion(out, label) + InfoNCE_features(device, v, t, 0.1)
                    loss_v = criterion(out_v, label)
                    loss_t = criterion(out_t, label)
                else:
                    raise NotImplementedError
                loss.backward()
        if args.use_tqdm:
            data_loader.set_description(f'Loss: {loss.cpu().item():.3f}')
     
        # modulation
        if args.dataset in AV_Datasets + VV_Datasets + Embed_Datasets:
            if args.modulation in ['OGM', 'OGM_GE']:
                OGM_modulation(args, model, out_a, out_v, label, epoch, logger)
            elif args.modulation == 'AGM':
                AGM_modulation(args, model, out_a, out_v, label, epoch, logger)
        elif args.dataset in VT_Datasets:
            if args.modulation in ['OGM', 'OGM_GE']:
                OGM_modulation(args, model, out_v, out_t, label, epoch, logger)
            elif args.modulation == 'AGM':
                AGM_modulation(args, model, out_v, out_t, label, epoch, logger)
        
        if args.modulation != 'MLA':       
            optimizer.step()
        
        if args.dataset in AV_Datasets + VV_Datasets + Embed_Datasets:
            logger.after_iter({'Loss/Total Loss':loss, 'Loss/Audio Loss':loss_a, 'Loss/Video Loss':loss_v})
        elif args.dataset in VT_Datasets:
            logger.after_iter({'Loss/Total Loss':loss, 'Loss/Video Loss':loss_v, 'Loss/Text Loss':loss_t})
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()
    logger.after_epoch()
    

def valid(args, model, device, dataloader, epoch, logger, save_feature=False):
    softmax = nn.Softmax(dim=1)
    n_classes = get_classes_num(args)
    logger.before_epoch(epoch)
    
    if save_feature:
        features = {
            'visual' : [], 
            'audio' : [],
            'text' : [],
            'OF' : [], 
            'label' : []
        }
        logits = {key:[] for key in features.keys()}
    
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = np.array([0.0 for _ in range(n_classes)])
        acc = np.array([0.0 for _ in range(n_classes)])
        acc_a = np.array([0.0 for _ in range(n_classes)])
        acc_v = np.array([0.0 for _ in range(n_classes)])
        acc_t = np.array([0.0 for _ in range(n_classes)])
        acc_mla = np.array([0.0 for _ in range(n_classes)])
        acc_qmf = np.array([0.0 for _ in range(n_classes)])
        
        if args.use_tqdm:
            dataloader = tqdm(dataloader, file=sys.stdout)
        else:
            dataloader = dataloader
        for step, batch in enumerate(dataloader):
            pred_a = pred_v = pred_t = None
            if args.dataset in AV_Datasets + VV_Datasets + Embed_Datasets:
                if args.dataset in AV_Datasets:
                    spec, image, label, idx = batch
                    spec = spec.to(device)
                    image = image.to(device)
                    label = label.to(device)
                    a, v, out = model(spec.unsqueeze(1).float(), image.float())
                    if save_feature:
                        features['audio'].append(a.detach().cpu())
                        features['visual'].append(v.detach().cpu())
                elif args.dataset in Embed_Datasets:
                    embed1 = batch['Embed1'].to(device)
                    embed2 = batch['Embed2'].to(device)
                    image = embed1
                    label = batch['label'].to(device)
                    a, v, out = model(embed1.float(), embed2.float())
                    if save_feature:
                        features['audio'].append(a.detach().cpu())
                        features['visual'].append(v.detach().cpu())
                else:
                    image1 = batch['images1'].to(device)
                    image2 = batch['images2'].to(device)
                    image = image1
                    label = batch['label'].to(device)
                    idx = batch['idx']
                    a, v, out = model(image1.float(), image2.float())
                    if save_feature:
                        features['visual'].append(a.detach().cpu())
                        features['OF'].append(v.detach().cpu())
                
                out_a = model.module.fusion_module.forward_a(a)
                out_v = model.module.fusion_module.forward_v(v)
                if save_feature:
                    logits['audio'].append(out_a.detach().cpu())
                    logits['visual'].append(out_v.detach().cpu())
                
                prediction = out
                pred_a = out_a
                pred_v = out_v
                mla_predictions = MLA_prediction(out_a, out_v)
                qmf_predicitons = QMF_prediciton(out_a, out_v)
            elif args.dataset in VT_Datasets:
                image = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)
                v, t, out = model(image.float(), (input_ids, attention_mask))
                if save_feature:
                    features['visual'].append(v.detach().cpu())
                    features['text'].append(t.detach().cpu())
                out_v = model.module.fusion_module.forward_a(v)
                out_t = model.module.fusion_module.forward_v(t)
                
                prediction = out
                pred_v = out_v
                pred_t = out_t
                mla_predictions = MLA_prediction(out_v, out_t)
                qmf_predicitons = QMF_prediciton(out_v, out_t)
                
            if save_feature:
                features['label'].append(label.detach().cpu())
                logits['label'].append(label.detach().cpu())
            
            pred_a = torch.zeros_like(prediction) if pred_a is None else pred_a
            pred_v = torch.zeros_like(prediction) if pred_v is None else pred_v
            pred_t = torch.zeros_like(prediction) if pred_t is None else pred_t
            
            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                t = np.argmax(pred_t[i].cpu().data.numpy())
                mla = np.argmax(mla_predictions[i].cpu().data.numpy())
                qmf = np.argmax(qmf_predicitons[i].cpu().data.numpy())
                num[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == t:
                    acc_t[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == mla:
                    acc_mla[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == qmf:
                    acc_qmf[label[i]] += 1.0
    print_num = min(len(num), 10)
    np.set_printoptions(precision=3)
    print("Audio acc : ", acc_a[:print_num] / num[:print_num])
    print("Video acc : ", acc_v[:print_num] / num[:print_num])
    print("Text acc : ", acc_t[:print_num] / num[:print_num])
    print("Total acc : ", acc[:print_num] / num[:print_num])
    
    logger.tb_writer_epoch('Acc/Acc', sum(acc) / sum(num))
    logger.tb_writer_epoch('Acc/Audio Acc', sum(acc_a) / sum(num))
    logger.tb_writer_epoch('Acc/Video Acc', sum(acc_v) / sum(num))
    logger.tb_writer_epoch('Acc/Text Acc', sum(acc_t) / sum(num))
    
    if save_feature:
        for key in features.keys():
            features[key] = torch.cat(features[key], dim=0) if len(features[key]) > 0 else None
            logits[key] = torch.cat(logits[key], dim=0) if len(logits[key]) > 0 else None
        return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num), sum(acc_mla) / sum(num), sum(acc_qmf) / sum(num), features, logits
    
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num), sum(acc_mla) / sum(num), sum(acc_qmf) / sum(num)


def build_model(args):
    if args.dataset in AV_Datasets:
        model = AVClassifier(args)
    elif args.dataset in VT_Datasets:
        model = VTClassifier(args)
    elif args.dataset in VV_Datasets:
        model = VVClassifier(args)
    elif args.dataset in Embed_Datasets:
        model = Embed_Classifier(args)
    return model


def main(args):
    setup_seed(args.random_seed)
    
    gpu_ids = args.gpu_ids if isinstance(args.gpu_ids, list) else [args.gpu_ids]
    gpu_ids = [ int(ids) for ids in gpu_ids]
    device = torch.device('cuda:{}'.format(gpu_ids[0]))
    
    model = build_model(args)
    # model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optim_param = []
    lr_audio = args.learning_rate if args.learning_rate_audio is None else args.learning_rate_audio
    lr_visual = args.learning_rate if args.learning_rate_visual is None else args.learning_rate_visual
    lr_text = args.learning_rate if args.learning_rate_text is None else args.learning_rate_text
    for key, value in dict(model.named_parameters()).items():
        if 'audio_net' in key:
            if args.weight_decay_audio is None:
                optim_param += [{'params': [value], 'lr': lr_audio}]
            else:
                optim_param += [{'params': [value], 'lr': lr_audio, 'weight_decay': args.weight_decay_audio}]
        elif 'visual_net' in key:
            if args.weight_decay_visual is None:
                optim_param += [{'params': [value], 'lr': lr_visual}]
            else:
                optim_param += [{'params': [value], 'lr': lr_visual, 'weight_decay': args.weight_decay_visual}]
        elif 'text_net' in key:
            if args.weight_decay_text is None:
                optim_param += [{'params': [value], 'lr': lr_text}]
            else:
                optim_param += [{'params': [value], 'lr': lr_text, 'weight_decay': args.weight_decay_text}]
        else:
            optim_param += [{'params': [value], 'lr': args.learning_rate}]
    
    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(  optim_param, 
                                lr=args.learning_rate, 
                                momentum=0.9, 
                                weight_decay=1e-4)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(optim_param, lr=args.learning_rate, betas = (0.9, 0.999), weight_decay=1e-4)
    
    # scheduler
    if args.scheduler == 'StepLR':    
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.scheduler == 'CosineLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) 
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    # scheduler load dataset
    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train', partition=(0, 0.8))
        val_dataset = VGGSound(args, mode='val', partition=(0.8, 1))
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticsSound':
        train_dataset = KineticsSound(args, mode='train', partition=(0, 0.8))
        val_dataset = KineticsSound(args, mode='val', partition=(0.8, 1))
        test_dataset = KineticsSound(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CREMADataset(args, mode='train', partition=(0, 0.8))
        val_dataset = CREMADataset(args, mode='val', partition=(0.8, 1))
        test_dataset = CREMADataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        val_dataset = AVEDataset(args, mode='val')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'UCF101':
        train_dataset = UCF101Dataset(args, mode='train', partition=(0, 0.8))
        val_dataset = UCF101Dataset(args, mode='val', partition=(0.8, 1))
        test_dataset = UCF101Dataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support CREMA-D, KineticsSound, AVE and VGGSound for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    
    logger = LOGGER(args)
    if args.loss_type == 'QMF':
        model.module.QMF_module.init_history(len(train_dataloader.dataset))

    if args.task == 'train':
        best_acc = 0.0
        best_acc_mla = 0.0
        best_acc_qmf = 0.0
        best_acc_audio = 0.0
        best_acc_visual = 0.0
        best_acc_text = 0.0
        
        if args.warm_up_epochs > 0:
            freeze_backbone(model)
        else:
            unfreeze_backbone(model)
            
        # print_trainable_weight(model)
        
        if args.loss_type == 'PMR':
            model.module.PMR_module.upate_prototypes(model, train_dataloader, device)

        for epoch in range(args.epochs):
            if args.warm_up_epochs > 0 and args.warm_up_epochs == epoch:
                unfreeze_backbone(model)
                # print_trainable_weight(model)

            print('\nEpoch: {}: '.format(epoch))

            train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler, logger=logger)
            if args.loss_type == 'PMR':
                model.module.PMR_module.upate_prototypes(model, train_dataloader, device)
            
            acc, acc_a, acc_v, acc_t, acc_mla, acc_qmf= valid(args, model, device, val_dataloader, epoch, logger)
            
            best_acc_mla = max(best_acc_mla, acc_mla)
            best_acc_qmf = max(best_acc_qmf, acc_qmf)
            best_acc_audio = max(best_acc_audio, acc_a)
            best_acc_visual = max(best_acc_visual, acc_v)
            best_acc_text = max(best_acc_text, acc_t)
            
            if acc > best_acc:
                best_acc = float(acc)
                if args.save_model:
                    if not os.path.exists(args.ckpt_path):
                        os.mkdir(args.ckpt_path)

                model_name = 'epoch_{}_acc_{:.3f}.pth'.format(epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'loss_type' : args.loss_type,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
                method_name = '{}_{}_{}'.format(args.modulation, args.fusion_method, args.loss_type)
                save_dir = os.path.join(args.ckpt_path, args.dataset, method_name, model_name)

                if args.save_model and not args.mute_save_middle_model:
                    create_dir(os.path.join(args.ckpt_path, args.dataset))
                    create_dir(os.path.join(args.ckpt_path, args.dataset, method_name))
                    torch.save(saved_dict, save_dir)
                    print('The best model has been saved at {}.'.format(save_dir))
            print("Acc: {:.3f}, Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f}".format(acc, acc_a, acc_v, acc_t))
            
        model.load_state_dict(saved_dict['model'])
        if args.save_model:
            create_dir(os.path.join(args.ckpt_path, args.dataset))
            create_dir(os.path.join(args.ckpt_path, args.dataset, method_name))
            method_name = '{}_{}_{}'.format(args.modulation, args.fusion_method, args.loss_type)
            model_name = 'best_model.pth'
            save_dir = os.path.join(args.ckpt_path, args.dataset, method_name, model_name)
            torch.save(saved_dict, save_dir)
            print('The best model has been saved at {}.'.format(save_dir))
        
        test_acc, test_acc_a, test_acc_v, test_acc_t, test_acc_mla, test_acc_qmf= valid(args, model, device, test_dataloader, epoch, logger)
        print("Final best Val Acc: {:.3f}, MLA Acc: {:.3f}, QMF Acc: {:.3f}".format(best_acc, best_acc_mla, best_acc_qmf))
        print("Final Test Acc: {:.3f}, MLA Acc: {:.3f}, QMF Acc: {:.3f}, Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f}".format(test_acc, test_acc_mla, test_acc_qmf, test_acc_a, test_acc_v, test_acc_t))
        best_acc, best_acc_mla, best_acc_qmf, best_acc_audio, best_acc_visual, best_acc_text = test_acc, test_acc_mla, test_acc_qmf, test_acc_a, test_acc_v, test_acc_t
    else:
        loaded_dict = torch.load(args.ckpt_path, map_location='cpu')
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion_method = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        loss_type = loaded_dict['loss_type']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        args.loss_type = loss_type
        args.modulation = modulation
        args.fusion_method = fusion_method
        # assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        # assert fusion_method == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        
        model = build_model(args)
        
        model.apply(weight_init)
        model.to(device)

        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        if args.save_feature_path:
            acc, acc_a, acc_v, acc_t, acc_mla, acc_qmf, features, logits= valid(args, model, device, test_dataloader, 0, logger, save_feature=True)
        else:
            acc, acc_a, acc_v, acc_t, acc_mla, acc_qmf= valid(args, model, device, test_dataloader, 0, logger)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}, accuracy_mla: {}'.format(acc, acc_a, acc_v, acc_mla))
        
        if args.save_feature_path:
            create_dir(os.path.join(args.save_feature_path, args.dataset))
            saved_dict = {'features': features, 'dataset': args.dataset, 'logits': logits}
            save_name = '{}_{}_{}_{}.pth'.format(fusion_method, modulation, loss_type, '.'.join((args.ckpt_path.split('/')[-1]).split('.')[:-1]))
            save_path = os.path.join(args.save_feature_path, args.dataset, save_name)
            torch.save(saved_dict, save_path)
        return acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t
    return best_acc, best_acc_mla, best_acc_qmf, best_acc_audio, best_acc_visual, best_acc_text        
    



random_seed = [
    3141,5926,5358,9793,2384,6264,3383,27950,288,4197,1693,9937,5105,8209,7494,4592,3078,1640,6286
]
t_table = {
    '95%' : [
        -1, 12.7065, 4.3026, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2621, 2.2282
    ],
    '99%' :[
        -1, 63.6551, 9.9247, 5.8408, 4.6041, 4.0322, 3.7074, 3.4995, 3.3554, 3.2498, 3.1693	
    ]
}

def update_Acc_dict(Acc_dict, acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t):
    Acc_dict['acc'].append(acc)
    Acc_dict['acc_mla'].append(acc_mla)
    Acc_dict['acc_qmf'].append(acc_qmf)
    Acc_dict['acc_a'].append(acc_a)
    Acc_dict['acc_v'].append(acc_v)
    Acc_dict['acc_t'].append(acc_t)
    return Acc_dict

def calculate_95_confidence_interval(Acc_dict):
    for key in Acc_dict.keys():
        acc = np.array(Acc_dict[key])
        num = acc.shape[0]
        mean = acc.mean()
        s = np.sqrt(((acc-mean)**2).sum() / (num-1))
        t = t_table['95%'][num-1] # 95%
        SE = s / math.sqrt(num)
        error = t * SE
        print(f"{key} mean: {mean}, error: {error}")

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    Acc_dict = {'acc' : [], 'acc_mla' : [], 'acc_qmf' : [], 'acc_a' : [], 'acc_v' : [], 'acc_t' : []}
    acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t = main(args)
    Acc_dict = update_Acc_dict(Acc_dict, acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t)
    if args.task=='train':
        print('\n\n##################################')
        print('##########Train finish!###########')
        print('##################################\n\n')
    
    if args.rounds > 1 and args.task=='train':
        args = args_mute_output(args)
        for round in range(args.rounds-1):
            args.__setattr__('random_seed', random_seed[round])
            acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t = main(args)
            Acc_dict = update_Acc_dict(Acc_dict, acc, acc_mla, acc_qmf, acc_a, acc_v, acc_t)
            
            print('\n\n##################################')
            print('##########Train finish!###########')
            print('##################################\n\n')
    
    print(Acc_dict)
    calculate_95_confidence_interval(Acc_dict)