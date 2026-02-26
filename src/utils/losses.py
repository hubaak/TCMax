import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict


def max_clip_loss(loss, max_loss = 10):
    if loss.item() > max_loss:
        s =  max_loss / loss
        loss *=  s.detach()
    return loss


# Multi-Label Learning from Single Positive Labels
def AN_LS_loss(logits, one_hot_labels, epsilon=0.1):
    pred = torch.sigmoid(logits)
    log_pred = torch.log(pred)
    neg_log_pred = torch.log(1-pred)
    loss = one_hot_labels * ((1-epsilon/2)*log_pred + (epsilon/2) * neg_log_pred)
    loss += (1-one_hot_labels) * ((1-epsilon/2)*neg_log_pred + (epsilon/2) * log_pred)
    return -loss.sum(dim=-1).mean()

# Simple and Robust Loss Design for Multi-Label Learning with Missing Labels
def Hill_loss(logits, one_hot_labels, _lambda=1.5, margin=1.0, gamma=2.0):
    logits_margin = logits - margin
    pred_pos = torch.sigmoid(logits_margin)
    pred_neg = torch.sigmoid(logits)
    
    pt = (1 - pred_pos) * one_hot_labels + (1 - one_hot_labels)
    focal_weight = pt ** gamma
    focal_weight
    
    los_pos = one_hot_labels * torch.log(pred_pos)
    los_neg = (1-one_hot_labels) * -(_lambda- pred_neg) * pred_neg ** 2
    loss = -(los_pos + los_neg)
    loss *= focal_weight
    
    return loss.mean()


# TCMax Loss
def TCMax_loss(args, model, a, v, label):
    B = a.shape[0]
    if args.fusion_method in ['sum', 'concat', 'share_head']:
        out_a = model.module.fusion_module.forward_a(a) # [B, C]
        out_v = model.module.fusion_module.forward_v(v) # [B, C]
        out_av_mat = out_a[:,None,:].repeat(1, B, 1) + out_v[None:,:].repeat(B, 1, 1) # [B, B, C]
    else:
        extend_a = a.view(B, 1, -1).repeat(1, B, 1) # [B, B, D]
        extend_v = v.view(1, B, -1).repeat(B, 1, 1) # [B, B, D]
        _, _, out_av_mat = model.module.fusion_module(extend_a.view(B*B, -1), extend_v.view(B*B, -1)) # [B*B, C]
        out_av_mat = out_av_mat.view(B, B, -1)
    
    prob_avl = torch.exp(out_av_mat-out_av_mat.mean())
    prob_avl = prob_avl / prob_avl.sum()
    idx = torch.arange(B, device=a.device)
    prob_avl_core = prob_avl[idx, idx, label]    
    prob_al_core = prob_avl.sum(dim=1)[idx, label]    
    prob_vl_core = prob_avl.sum(dim=0)[idx, label]   
    loss = - torch.log(prob_avl_core).mean() - math.log(B)
    loss_a = - torch.log(prob_al_core).mean() - math.log(B)
    loss_v = - torch.log(prob_vl_core).mean() - math.log(B)
    
    return loss, loss_a, loss_v

def TCMax_loss_sample(args, model, a, v, label, n_classes):
    B = a.shape[0]
    sample_num = args.TCMax_sample_num
    if args.fusion_method in ['sum', 'concat', 'share_head']:
        out_a = model.module.fusion_module.forward_a(a) # [B, C]
        out_v = model.module.fusion_module.forward_v(v) # [B, C]
        out_pos = out_a + out_v
        if sample_num <= 0:
            random_indices = torch.randint(0, B, (B,))
            out_neg = out_a + out_v[random_indices]
        else:
            random_indices1 = torch.randint(0, B, (sample_num,))
            random_indices2 = torch.randint(0, B, (sample_num,))
            out_neg = out_a[random_indices1] + out_v[random_indices2]
    else:
        out_a = model.module.fusion_module.forward_a(a) # [B, C]
        out_v = model.module.fusion_module.forward_v(v) # [B, C]
        out_pos = model.module.fusion_module(a, v)
        if sample_num <= 0:
            random_indices = torch.randint(0, B, (B,))
            extend_a = a
            extend_v = v[random_indices]
        else:
            random_indices1 = torch.randint(0, B, (sample_num,))
            random_indices2 = torch.randint(0, B, (sample_num,))
            extend_a = a[random_indices1]
            extend_v = v[random_indices2]
        _, _, out_neg = model.module.fusion_module(extend_a, extend_v)
    
    idx = torch.arange(B, device=a.device)
    out_pos = out_pos[idx, label]
    mean = out_neg.mean()
    loss = - (out_pos.mean()-mean) + torch.log((torch.exp(out_neg - mean)).mean())
    
    loss = loss + math.log(B) + math.log(n_classes)
    loss_a = F.cross_entropy(out_a, label)
    loss_v = F.cross_entropy(out_v, label)
    
    return loss, loss_a, loss_v



def InfoNCE_features(device, a, v, tau=0.1, if_norm=True):
    if if_norm:
        a = a / a.norm(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
    
    logits = torch.matmul(a, v.T) / tau
    labels = torch.arange(a.shape[0]).to(device)
    loss_a = F.cross_entropy(logits, labels)
    loss_v = F.cross_entropy(logits.T, labels)
    return (loss_a+loss_v)*0.5


def Unimodal_loss(args, out_a, out_v, label, criterion):
    loss_a = criterion(out_a, label)
    loss_v = criterion(out_v, label)
    loss = loss_a + loss_v
    return loss, loss_a, loss_v


def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target.reshape(-1,1))

    return ranking_loss


def QMF_loss(model, out_a, out_v, label, idx):
    audio_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
    img_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

    audio_conf = audio_energy / 10
    img_conf = img_energy / 10
    audio_conf = torch.reshape(audio_conf, (-1, 1))
    img_conf = torch.reshape(img_conf, (-1, 1))
    out = (out_a * audio_conf.detach() + out_v * img_conf.detach())

    audio_clf_loss = nn.CrossEntropyLoss()(out_a, label)
    img_clf_loss = nn.CrossEntropyLoss()(out_v, label)
    # clf_loss = audio_clf_loss + img_clf_loss
    clf_loss=audio_clf_loss + img_clf_loss

    audio_loss = nn.CrossEntropyLoss(reduction='none')(out_a, label).detach()
    img_loss = nn.CrossEntropyLoss(reduction='none')(out_v, label).detach()

    model.module.QMF_module.audio_history.correctness_update(idx, audio_loss, audio_conf.squeeze())
    model.module.QMF_module.visual_history.correctness_update(idx, img_loss, img_conf.squeeze())

    audio_rank_loss = rank_loss(audio_conf, idx, model.module.QMF_module.audio_history)
    img_rank_loss = rank_loss(img_conf, idx, model.module.QMF_module.visual_history)

    crl_loss = audio_rank_loss + img_rank_loss
    cml_loss = nn.CrossEntropyLoss()(out, label)
                    
    loss = cml_loss + clf_loss + crl_loss
    
    return loss, audio_clf_loss, img_clf_loss





def OPM_loss(args, model, a, v, out_a, out_v, label, device):
    B = out_a.shape[0]
    if 'q_a' not in OPM_loss.__dict__:
        OPM_loss.q_a = 0.0
        OPM_loss.q_v = 0.0
        
    score_v = sum([F.softmax(out_v, dim=1)[i][label[i]] for i in range(out_v.size(0))]).detach()
    score_a = sum([F.softmax(out_a, dim=1)[i][label[i]] for i in range(out_a.size(0))]).detach()
    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v
    
    q_a = args.OPM_qbase * (1 + args.OPM_lambda * torch.tanh(ratio_a)) if ratio_a > 1 else 0
    q_v = args.OPM_qbase * (1 + args.OPM_lambda * torch.tanh(ratio_v)) if ratio_v > 1 else 0
    
    mask_a = torch.rand(B, device=device) < OPM_loss.q_a
    mask_v = torch.rand(B, device=device) < OPM_loss.q_v
    
    a[mask_a] = 0
    v[mask_v] = 0
    
    _, _, out = model.module.fusion_module(a, v)
    
    loss = nn.CrossEntropyLoss()(out, label)
    loss_a = nn.CrossEntropyLoss()(out_a, label)
    loss_v = nn.CrossEntropyLoss()(out_v, label)
    
    OPM_loss.q_a = q_a.detach() if isinstance(q_a, torch.Tensor) else q_a
    OPM_loss.q_v = q_v.detach() if isinstance(q_v, torch.Tensor) else q_v
    
    return loss, loss_a, loss_v


# ##################### MCR #####################
    
# def js_divergence(net_1_logits, net_2_logits):
#     clip_value = 1e7

#     net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
#     net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

#     total_m = 0.5 * (net_1_probs + net_2_probs)

#     clip_value = 1e-20
#     total_m = torch.clamp(total_m, clip_value, 1).log()

#     # total_m = torch.clamp(total_m, clip_value, 1)
#     net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
#     net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

#     loss = 0.0

#     loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
#     loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
#     if torch.isnan(loss):
#         raise Exception("NaN detected in loss computation")
#     return 0.5 * loss

# def log_likelihood_ratio(logits_Z, logits_Z_hat):
#     """
#     Compute the negative average log likelihood ratio between two sets of logits.

#     Args:
#     - logits_Z (torch.Tensor): Logits from predictions using (X, Z), shape (batch_size, num_classes)
#     - logits_Z_hat (torch.Tensor): Logits from predictions using (X, Z_hat), shape (batch_size, num_classes)
#     - targets (torch.Tensor): True class labels, shape (batch_size,)

#     Returns:
#     - torch.Tensor: Difference of negative average log likelihood ratios
#     """
#     # Compute probabilities from logits

#     probs_Z = F.softmax(logits_Z, dim=1)
#     probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

#     # label_norm = targets.repeat(len(pert_data))[ids_of_each_sample]

#     # torch.concatenate([targets[pert_data[i]["data"]] for i in range(len(pert_data))], dim=0)

#     # Select the probabilities corresponding to the true labels
#     # target_probs_Z_hat = probs_Z_hat[range(len(label_norm)), label_norm]
#     # target_probs_Z = probs_Z[range(len(targets)), targets]

#     # Compute log probabilities
#     # log_probs_Z = torch.log(target_probs_Z + 1e-9)  # Add epsilon to avoid log(0)
#     # log_probs_Z_hat = torch.log(target_probs_Z_hat + 1e-9)  # Add epsilon to avoid log(0)

#     # Compute log probabilities
#     # log_probs_Z = torch.log(probs_Z + 1e-7)  # Add epsilon to avoid log(0)
#     # log_probs_Z_hat = torch.log(probs_Z_hat + 1e-7)  # Add epsilon to avoid log(0)
#     log_probs_Z = F.log_softmax(logits_Z, dim=1)
#     log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

#     per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
#     per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

#     #     print(log_probs_Z[ids_of_each_sample==i].mean())
#     # # Compute the separate empirical averages
#     # avg_log_prob_Z = torch.concat([log_probs_Z[ids_of_each_sample==i].mean() for i in len(pert_data[0]["data"])])
#     avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
#     avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

#     avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
#     avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

#     # avg_log_prob_Z_hat = (per_sample_avg_log_prob_Z * per_sample_avg_prob_Z).mean()

#     # Compute the difference of the negative average log likelihood ratios
#     difference = -avg_log_prob_Z + avg_log_prob_Z_hat
#     difference_onlylog = -avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

#     return (
#         difference_onlylog,
#         avg_log_prob_Z_hat_onlylog,
#         avg_log_prob_Z_onlylog,
#     )

# def get_losses(
#     predicted_logits_00,
#     soft_labels_norm,
#     cross_permod,
#     label,
#     label_shuffled,
#     title,
# ):

#     jsd_label = js_divergence(predicted_logits_00, label)
#     jsd_pred = js_divergence(predicted_logits_00, soft_labels_norm)
#     loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(soft_labels_norm, predicted_logits_00)

#     cross_permod["{}_yent".format(title)].append(entropy_y)
#     cross_permod["{}_nll".format(title)].append(loglike_pred)
#     cross_permod["{}_label".format(title)].append(jsd_label)
#     cross_permod["{}_pred".format(title)].append(jsd_pred)

#     return cross_permod

# def MCR_loss(model, a, v, label, num_classes=6):
#     output = model.module.fusion_module.forward_train(a, v, label=label)
#     label = F.one_hot(
#         output["preds"]["n_label"],
#         num_classes=num_classes,
#     )
#     label_shuffled = F.one_hot(
#         output["preds"]["n_label_shuffled"],
#         num_classes=num_classes,
#     )
#     cross_permod = defaultdict(list)
#     cross_permod = get_losses(
#         output["preds"]["sa_detv"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="enc0_p0",
#     )
#     cross_permod = get_losses(
#         output["preds"]["sa_deta"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="enc1_p0",
#     )
#     cross_permod = get_losses(
#         output["preds"]["sa"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="all_p0",
#     )
    
#     cross_permod = get_losses(
#         output["preds"]["sv_deta"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="enc1_p1",
#     )
#     cross_permod = get_losses(
#         output["preds"]["sv_detv"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="enc0_p1",
#     )
#     cross_permod = get_losses(
#         output["preds"]["sv"],
#         output["preds"]["ncombined"],
#         cross_permod,
#         label=label,
#         label_shuffled=label_shuffled,
#         title="all_p1",
#     )
#     out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}
#     print(out)
    
    
    
# ##################### MCR End #####################