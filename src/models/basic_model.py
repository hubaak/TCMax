import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18, resnet34, resnet50, load_pretrained
from .fusion_modules import *
from .MLA_module import MLA_GSPlugin
from .PMR_module import PMR_module
from .AGM_module import AGM_module
from .QMF_module import QMF_module
from transformers import DistilBertModel, BertModel
from clip import clip
from .transformer import TextTransformer
import torchvision.models as TV_models
from torchvision.models import vision_transformer, VisionTransformer
    
def get_classes_num(args):
    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticsSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'IEMOCAP':
        n_classes = 4
    elif args.dataset == 'IEMOCAP_4':
        n_classes = 4
    elif args.dataset == 'IEMOCAP_6':
        n_classes = 6
    elif args.dataset in ['MVSA', 'MVSA_Embed']:
        n_classes = 3
    elif args.dataset in ['Food101', 'Food101_Embed']:
        n_classes = 101
    elif args.dataset == 'UCF101':
        n_classes = 101
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
    return n_classes

def get_fusion_module(args, embed_dim = 512):
    n_classes = get_classes_num(args)
    fusion = args.fusion_method
    if fusion == 'sum':
        fusion_module = SumFusion(input_dim=embed_dim, output_dim=n_classes)
    elif fusion == 'concat':
        fusion_module = ConcatFusion(input_dim=embed_dim*2, output_dim=n_classes, args=args)
    elif fusion == 'CLIP':
        fusion_module = CLIP_Predict_Head(input_dim=embed_dim, output_dim=n_classes)
    elif fusion == 'concat_msh':
        fusion_module = ConcatFusion_with_modality_specific_head(input_dim=embed_dim, output_dim=n_classes, args=args)
    elif fusion == 'film':
        fusion_module = FiLM(input_dim=embed_dim, output_dim=n_classes, x_film=True)
    elif fusion == 'gated':
        fusion_module = GatedFusion(input_dim=embed_dim, output_dim=n_classes, x_gate=True)
    elif fusion == 'bigated':
        fusion_module = BiGateFusion(input_dim=embed_dim, output_dim=n_classes)
    elif fusion == 'share_head':
        fusion_module = Share_Head(input_dim=embed_dim, output_dim=n_classes)
    else:
        raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
    return fusion_module


def get_audio_model(args):
    if args.audio_backbone == 'RN18':
        audio_net = resnet18(modality='audio', args=args)
        audio_embed_dim = 512
    
    if args.audio_weight is not None:
        loaded_dict = torch.load(args.audio_weight, map_location='cpu')
        state_dict = loaded_dict['model']
        state_dict = {k.replace('module.audio_net.', ''): v for k, v in state_dict.items() if 'audio_net' in k}
        audio_net.load_state_dict(state_dict)
        
        # freeze the model
        for param in audio_net.parameters():
            param.requires_grad = False
    
    return audio_net, audio_embed_dim

def get_visual_model(args, modality='visual'):
    if args.visual_backbone == 'RN18':
        visual_net = resnet18(modality=modality, args=args)
        visual_embed_dim = 512
    elif args.visual_backbone == 'RN18_pretrained':
        pretrained_model = TV_models.resnet18(pretrained=True)
        visual_net = resnet18(modality=modality, args=args)
        load_pretrained(visual_net, pretrained_model, modality)
        visual_embed_dim = 512
    elif args.visual_backbone == 'RN34_pretrained':
        pretrained_model = TV_models.resnet34(pretrained=True)
        visual_net = resnet34(modality=modality, args=args)
        load_pretrained(visual_net, pretrained_model, modality)
        visual_embed_dim = 512
    elif args.visual_backbone == 'RN50_pretrained':
        pretrained_model = TV_models.resnet50(pretrained=True)
        visual_net = resnet50(modality=modality, args=args)
        load_pretrained(visual_net, pretrained_model, modality)
        visual_embed_dim = 2048
    elif args.visual_backbone == 'CLIP_RN50':
        visual_net = clip.load('RN50', device="cpu")[0]
        visual_net.pretrained_names = [k for k, v in visual_net.state_dict().items()]
        visual_embed_dim = 1024
    elif args.visual_backbone == 'ViT_B_32':
        visual_net = VisionTransformer(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072
        )
        visual_embed_dim = 768
    elif args.visual_backbone == 'ViT_B_16':
        visual_net = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072
        )
        visual_embed_dim = 768
    elif 'CLIP_' in args.visual_backbone:
        visual_net = None
        visual_embed_dim = 512
        if args.visual_backbone == 'CLIP_ViT-B/16':
            visual_embed_dim = 512
    else:
        raise NotImplementedError
    
    
    if args.visual_weight is not None:
        loaded_dict = torch.load(args.visual_weight, map_location='cpu')
        state_dict = loaded_dict['model']
        state_dict = {k.replace('module.visual_net.', ''): v for k, v in state_dict.items() if 'visual_net' in k}
        visual_net.load_state_dict(state_dict)
        
        # freeze the model
        for param in visual_net.parameters():
            param.requires_grad = False
    
    return visual_net, visual_embed_dim

def get_text_model(args):
    if args.text_backbone == 'Bert':
        text_net = BertModel.from_pretrained('/data/wuxy/paper/Models/bert-base-uncased')
        text_embed_dim = 768
    elif args.text_backbone == 'DistilBert':
        text_net = DistilBertModel.from_pretrained('/data/wuxy/paper/Models/distilbert-base-uncased')
        text_embed_dim = 768
    elif args.text_backbone == 'CLIP_RN50':
        text_net = clip.load('RN50', device="cpu")[0]
        text_net.pretrained_names = [k for k, v in text_net.state_dict().items()]
        text_embed_dim = 1024
    elif args.text_backbone == 'transformer':
        text_net = TextTransformer(512, 77, 49408, 512, 8, 12)
        text_embed_dim = 512
    else:
        raise NotImplementedError
    return text_net, text_embed_dim
        
    
class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()
        self.args = args
        self.n_classes = get_classes_num(args)
        self.fusion_module = get_fusion_module(args, embed_dim = 512)
        self.pretrained_names = []

        self.audio_net, audio_embed_dim = get_audio_model(args)
        self.visual_net, visual_embed_dim = get_visual_model(args)
        
        if args.modulation == 'MLA':
            self.mla_gs_plugin = MLA_GSPlugin()
        if args.modulation == 'AGM':
            self.AGM_module = AGM_module(args)
        if args.loss_type == 'PMR':
            self.PMR_module = PMR_module(args, 512, self.n_classes, 2)
        if args.loss_type == 'QMF':
            self.QMF_module = QMF_module(args)

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out
    
    def encode_audio(self, audio):
        a = self.audio_net(audio)
        B = a.size()[0]
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        return a
    
    def encode_video(self, visual):
        v = self.visual_net(visual)
        (B, C, H, W) = v.size()
        B = B // self.args.frames
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v
    
    def forward_vision(self, visual):
        visual = visual[:, :, None]
        v = self.visual_net(visual)
        (B, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.fusion_module.forward_v(v)
        return out

    
        
class VTClassifier(nn.Module):
    def __init__(self, args):
        super(VTClassifier, self).__init__()
        self.args = args
        self.n_classes = get_classes_num(args)
        self.pretrained_names = []

        self.text_net, text_embed_dim = get_text_model(args)
        self.visual_net, visual_embed_dim = get_visual_model(args)
        
        if text_embed_dim != visual_embed_dim:
            self.text_adapter = nn.Linear(text_embed_dim, visual_embed_dim, bias=True)
        else:
            self.text_adapter = nn.Identity()
        self.embed_dim = visual_embed_dim
        
        if hasattr(self.visual_net, 'pretrained_names'):
            self.pretrained_names += ['visual_net.'+name for name in self.visual_net.pretrained_names]
        if hasattr(self.text_net, 'pretrained_names'):
            self.pretrained_names += ['text_net.'+name for name in self.text_net.pretrained_names]
        
        self.fusion_module = get_fusion_module(args, embed_dim = self.embed_dim)
        
        if args.modulation == 'MLA':
            self.mla_gs_plugin = MLA_GSPlugin()
        if args.modulation == 'AGM':
            self.AGM_module = AGM_module(args)
        if args.loss_type == 'PMR':
            self.PMR_module = PMR_module(args, self.embed_dim, self.n_classes, 2)
        if args.loss_type == 'QMF':
            self.QMF_module = QMF_module(args)

    def forward(self, visual, text):
        v = self.encode_video(visual)
        t = self.encode_text(text[0], text[1])
        v, t, out = self.fusion_module(v, t)
        return v, t, out
    
    def encode_text(self, input_ids, attention_mask):
        if 'CLIP' in self.args.text_backbone:
            outputs = self.text_net.encode_text(input_ids)
        else:
            outputs = self.text_net(input_ids, attention_mask=attention_mask)
            outputs = outputs.last_hidden_state[:, 0, :] # [CLS] token
        return self.text_adapter(outputs)
    
    def encode_video(self, visual):
        if 'CLIP' in self.args.visual_backbone:
            v = self.visual_net.encode_image(visual[:, :, 0])
        else:
            v = self.visual_net(visual)
            (B, C, H, W) = v.size()
            B = B
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)
        return v
    
    
    
class VVClassifier(nn.Module):
    def __init__(self, args):
        super(VVClassifier, self).__init__()
        self.args = args
        self.n_classes = get_classes_num(args)
        
        self.pretrained_names = []
        self.visual_net1, visual_embed_dim = get_visual_model(args, modality='visual')
        self.visual_net2, _ = get_visual_model(args, modality='optic_flow')
        if hasattr(self.visual_net1, 'pretrained_names'):
            self.pretrained_names += ['visual_net1'+name for name in self.visual_net1.pretrained_names]
        if hasattr(self.visual_net2, 'pretrained_names'):
            self.pretrained_names += ['visual_net2'+name for name in self.visual_net2.pretrained_names]
        
        self.fusion_module = get_fusion_module(args, embed_dim = visual_embed_dim)
        
        if args.modulation == 'MLA':
            self.mla_gs_plugin = MLA_GSPlugin()
        if args.modulation == 'AGM':
            self.AGM_module = AGM_module(args)
        if args.loss_type == 'PMR':
            self.PMR_module = PMR_module(args, 512, self.n_classes, 2)
        if args.loss_type == 'QMF':
            self.QMF_module = QMF_module(args)

    def forward(self, visual1, visual2):
        v1 = self.encode_video1(visual1)
        v2 = self.encode_video2(visual2)

        v1, v2, out = self.fusion_module(v1, v2)

        return v1, v2, out
    
    def encode_video1(self, visual):
        v = self.visual_net1(visual)
        (B, C, H, W) = v.size()
        B = B // self.args.frames
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v
    
    def encode_video2(self, visual):
        v = self.visual_net2(visual)
        (B, C, H, W) = v.size()
        # B = B // self.args.OF_frames
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v
    
    

class Embed_Classifier(nn.Module):
    def __init__(self, args):
        super(Embed_Classifier, self).__init__()
        self.args = args
        self.n_classes = get_classes_num(args)
        self.pretrained_names = []
        _, visual_embed_dim = get_visual_model(args, modality='visual')
        self.fusion_module = get_fusion_module(args, embed_dim = visual_embed_dim)
        if args.modulation == 'MLA':
            self.mla_gs_plugin = MLA_GSPlugin()
        if args.modulation == 'AGM':
            self.AGM_module = AGM_module(args)
        if args.loss_type == 'PMR':
            self.PMR_module = PMR_module(args, 512, self.n_classes, 2)
        if args.loss_type == 'QMF':
            self.QMF_module = QMF_module(args)
            
    def forward(self, embed1, embed2):
        _, _, out = self.fusion_module(embed1, embed2)
        return embed1, embed2, out