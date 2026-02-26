import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

AV_Datasets = ['CREMAD', 'KineticsSound', 'AVE', 'VGGSound']
VT_Datasets = ['MVSA', 'Food101']
AVT_Datasets  = ['IEMOCAP']
VV_Datasets = ['UCF101']


def max_clip_loss(loss, max_loss = 10):
    if loss.item() > max_loss:
        s =  max_loss / loss
        loss *=  s.detach()
    return loss

def max_ignore_loss(loss, max_loss = 10):
    if loss.item() > max_loss:
        s =  max_loss / loss
        loss = 0
    return loss


class PMR_module(nn.Module):
    def __init__(self, args, embed_dim, n_classes, modality_num = 2):
        super(PMR_module, self).__init__()
        self.args = args
        self.n_classes = n_classes
        self.alpha = args.PMR_alpha
        self.epsilon = args.PMR_epsilon
        self.mu = args.PMR_mu
        self.to_be_init = True
        self.prototypes = torch.zeros([modality_num, n_classes, embed_dim])
    
    @torch.no_grad()
    def feed_forword_get_prototypes(self, model, dataloader, device):
        model.eval()    
        sample_count = 0
        all_num = len(dataloader)
        count_class = torch.zeros(self.n_classes).to(device)
        new_prototypes = torch.zeros_like(self.prototypes).to(device)
        for _, batch in enumerate(dataloader):
            if self.args.dataset in AV_Datasets + VV_Datasets + VT_Datasets:
                if self.args.dataset in AV_Datasets:
                    spec, image, label, idx = batch
                    spec = spec.to(device)
                    image = image.to(device)
                    label = label.to(device)
                    a, v, _ = model(spec.unsqueeze(1).float(), image.float())
                elif self.args.dataset in VV_Datasets:
                    image1 = batch['images1'].to(device)
                    image2 = batch['images2'].to(device)
                    label = batch['label'].to(device)
                    idx = batch['idx']
                    a, v, out = model(image1.float(), image2.float())
                elif self.args.dataset in VT_Datasets:
                    image = batch['images'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    label = batch['label'].to(device)
                    idx = batch['idx']
                    a, v, out = model(image.float(), (input_ids, attention_mask))
                    
                for c, l in enumerate(label):
                    count_class[l] += 1
                    new_prototypes[0, l, :] += a[c, :]
                    new_prototypes[1, l, :] += v[c, :]
                sample_count += 1
                if self.args.dataset == 'AVE':
                    pass 
                else:
                    if sample_count >= all_num * 0.1:
                        break
            elif self.args.dataset in ['IEMOCAP']:
                spec = batch['spectrogram'].to(device)
                image = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)
                a, v, t, _ = model(spec.unsqueeze(1).float(), image.float(), (input_ids, attention_mask))
                for c, l in enumerate(label):
                    count_class[l] += 1
                    new_prototypes[0, l, :] += a[c, :]
                    new_prototypes[1, l, :] += v[c, :]
                    new_prototypes[2, l, :] += t[c, :]
                    sample_count += 1
            else:
                raise NotImplementedError(f"{self.args.datase} dataset is not support")

        new_prototypes /= count_class[None, :, None]
        return new_prototypes
    
    def upate_prototypes(self, model, dataloader, device):
        self.device = self.prototypes.device
        new_prototypes = self.feed_forword_get_prototypes(model, dataloader, device).to(self.device)
        if self.to_be_init:
            self.prototypes = new_prototypes
        else:
            self.prototypes = self.epsilon * self.prototypes + (1-self.epsilon) * new_prototypes
    
    def EU_dist(self, x1, x2):
        d_matrix = x1[:, None, :] - x2[None, :, :]
        d_matrix = d_matrix.norm(dim=-1)
        return d_matrix
    
    def PMR_loss(self, out, a, v, label, criterion, device, logger):
        sim_a = -self.EU_dist(a, self.prototypes[0].to(device)) # [B, n_class]
        sim_v = -self.EU_dist(v, self.prototypes[1].to(device)) # [B, n_class]
        score_a_p = F.softmax(sim_a, dim=-1)[torch.arange(0, a.shape[0]), label].sum().detach()
        score_v_p = F.softmax(sim_v, dim=-1)[torch.arange(0, v.shape[0]), label].sum().detach()
        
        loss_proto_a = criterion(sim_a, label)
        loss_proto_v = criterion(sim_v, label)
        
        min_score = min([score_a_p, score_v_p])
        ratio_a = score_a_p / min_score
        ratio_v = score_v_p / min_score
        ratio_a_p = score_a_p / score_v_p
        if ratio_a_p > 1:
            beta = 0  # audio coef
            lam = min(ratio_a_p-1.0, 1.0) # visual coef
        elif ratio_a_p < 1:
            beta = min(1.0/ratio_a_p-1.0, 1.0)
            lam = 0
        else:
            beta = 0
            lam = 0
        
        logger.statistics_add('PMR/ratio_a', ratio_a)
        logger.statistics_add('PMR/ratio_v', ratio_v)
        logger.statistics_add('PMR/ratio_a_p', ratio_a_p)
        
        L_ce = criterion(out, label)
        L_acc =  L_ce + self.alpha * beta * max_ignore_loss(loss_proto_a) + self.alpha * lam * max_ignore_loss(loss_proto_v)
        # L_acc =  L_ce + self.alpha * beta * max_clip_loss(loss_proto_a) + self.alpha * lam * max_clip_loss(loss_proto_v)
        
        # PER 
        # There is code for this part in the original repo: https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR
        # And the Equation 13 in the paper of PMR is unclear (and even wrong as both entropy is calculated by features from the same modality in the equation)
        # , so I produce the code with my own understanding
        score_label_2_a = F.softmax(sim_a, dim=0)
        score_label_2_v = F.softmax(sim_v, dim=0)
        entropy_a = - (score_label_2_a * torch.log(score_label_2_a)).sum(dim=0).mean()
        entropy_v = - (score_label_2_v * torch.log(score_label_2_v)).sum(dim=0).mean()
        
        L_final = L_acc - self.mu * lam * entropy_a - self.mu * beta * entropy_v
        
        return L_final, loss_proto_a, loss_proto_v
    
    def PMR_loss_AVT(self, out, a, v, t, label, criterion, device, logger):
        # TODO
        sim_a = -self.EU_dist(a, self.prototypes[0].to(device)) # [B, n_class]
        sim_v = -self.EU_dist(v, self.prototypes[1].to(device)) # [B, n_class]
        sim_t = -self.EU_dist(t, self.prototypes[2].to(device)) # [B, n_class]
        score_a_p = F.softmax(sim_a, dim=-1)[torch.arange(0, a.shape[0]), label].sum()
        score_v_p = F.softmax(sim_v, dim=-1)[torch.arange(0, v.shape[0]), label].sum()
        score_t_p = F.softmax(sim_t, dim=-1)[torch.arange(0, t.shape[0]), label].sum()
        
        loss_proto_a = criterion(sim_a, label)
        loss_proto_v = criterion(sim_v, label)
        loss_proto_t = criterion(sim_t, label)
        
        min_score = min([score_a_p, score_v_p, score_t_p])
        ratio_a = score_a_p / min_score
        ratio_v = score_v_p / min_score
        ratio_t = score_t_p / min_score
        beta = min(ratio_a-1, 1)
        gamma = min(ratio_v-1, 1)
        zeta = min(ratio_t-1, 1)
        logger.statistics_add('PMR/ratio_a', ratio_a)
        logger.statistics_add('PMR/ratio_v', ratio_v)
        logger.statistics_add('PMR/ratio_t', ratio_t)
        
        L_ce = criterion(out, label)
        L_acc =  L_ce + self.alpha * beta * loss_proto_a + self.alpha * gamma * loss_proto_v + self.alpha * zeta * loss_proto_t
        
        # PER TODO
        return L_acc, loss_proto_a, loss_proto_v, loss_proto_t