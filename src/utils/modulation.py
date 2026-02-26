import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np

AV_Datasets = ['CREMAD', 'KineticsSound', 'AVE', 'VGGSound']
VT_Datasets = ['MVSA', 'Food101']
AVT_Datasets  = ['IEMOCAP', 'IEMOCAP_4', 'IEMOCAP_6']
VV_Datasets = ['UCF101']

def OGM_modulation(args, model, out_a, out_v, label, epoch, logger):
    score_v = sum([F.softmax(out_v, dim=1)[i][label[i]] for i in range(out_v.size(0))])
    score_a = sum([F.softmax(out_a, dim=1)[i][label[i]] for i in range(out_a.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v
    logger.statistics_add('OGM/ratio_a', ratio_a)
    if args.modulation_starts <= epoch <= args.modulation_ends:
        if ratio_v > 1:
            coeff_v = 1 - F.tanh(args.OGM_alpha * F.relu(ratio_v))
            coeff_a = 1
        else:
            coeff_a = 1 - F.tanh(args.OGM_alpha * F.relu(ratio_a))
            coeff_v = 1
        for name, parms in model.named_parameters():
            if not parms.requires_grad:
                continue
            layer = str(name).split('.')[1]
            if args.dataset in AV_Datasets:
                if 'audio' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':
                        parms.grad = parms.grad * coeff_a + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                            parms.grad *= coeff_a

                if 'visual' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':
                        parms.grad = parms.grad * coeff_v + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_v
            elif args.dataset in VV_Datasets:
                if 'visual_net1' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':
                        parms.grad = parms.grad * coeff_a + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                            parms.grad *= coeff_a

                if 'visual_net2' in layer and len(parms.grad.size()) == 4:
                    if args.modulation == 'OGM_GE':
                        parms.grad = parms.grad * coeff_v + \
                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_v
            else:
                raise NotImplementedError


def MLA_modulation(args, model, criterion, optimizer, a, v, label, batch_step, len_dataloader):
    out_a = model.module.fusion_module.forward_a(a)
    loss_a = criterion(out_a, label)
    loss_a.backward()
    model.module.mla_gs_plugin.before_update(model.module.fusion_module.fc_out, a, 
                                    batch_step, len_dataloader,  model.module.mla_gs_plugin.exp_count)
    optimizer.step()
    optimizer.zero_grad()
    model.module.mla_gs_plugin.exp_count += 1
            
    out_v = model.module.fusion_module.forward_v(v)
    loss_v = criterion(out_v, label)
    loss_v.backward()
    model.module.mla_gs_plugin.before_update(model.module.fusion_module.fc_out, v, 
                            batch_step, len_dataloader,  model.module.mla_gs_plugin.exp_count)
    optimizer.step()
    optimizer.zero_grad()
    model.module.mla_gs_plugin.exp_count += 1
    
    return loss_a+loss_v, loss_a, loss_v  

def MLA_prediction(out_a, out_v):
    out_list = torch.cat([out_a[None,:], out_v[None,:]])
    softmax_n1 = nn.Softmax(dim=-1)
    softmax_0 = nn.Softmax(dim=0)
    predict = softmax_n1(out_list)
    entropy = - predict * torch.log(predict) 
    entropy = entropy.sum(dim=-1, keepdims=True) 
    lambda_mr = - entropy
    out = softmax_0(lambda_mr) * out_list
    return out.sum(dim=0)


def AGM_modulation(args, model, out_a, out_v, label, epoch, logger):
    if torch.isnan(out_a).any() or torch.isnan(out_v).any():
        raise ValueError
    
    idx = torch.arange(0, label.shape[0]).to(label.device)
    score_audio = torch.clamp(F.softmax(out_a, dim=-1), min=1e-8)
    score_audio = -torch.log(score_audio[idx, label]).mean()
    score_visual = torch.clamp(F.softmax(out_v, dim=-1), min=1e-8)
    score_visual = -torch.log(score_visual[idx, label]).mean()
    
    ratio_a = math.exp(score_visual.item() - score_audio.item())
    ratio_v = math.exp(score_audio.item() - score_visual.item())
    logger.statistics_add('AGM/ratio_a', ratio_a)
    logger.statistics_add('AGM/ratio_v', ratio_v)
    
    if args.modulation_starts <= epoch <= args.modulation_ends:
        train_score_a = model.module.AGM_module.train_score_a
        train_score_v = model.module.AGM_module.train_score_v
        
        optimal_ratio_a = math.exp(train_score_v - train_score_a)
        optimal_ratio_v = math.exp(train_score_a - train_score_v)
        
        coeff_a = math.exp(args.AGM_alpha*(min(optimal_ratio_a - ratio_a,10)))
        coeff_v = math.exp(args.AGM_alpha*(min(optimal_ratio_v - ratio_v,10)))
        
        model.module.AGM_module.updata_train_score(score_audio, score_visual)
        
        for name, parms in model.named_parameters():
            if not parms.requires_grad:
                continue
            layer = str(name).split('.')[1]
            if args.dataset in AV_Datasets:
                if 'audio' in layer:
                    parms.grad *= coeff_a
                if 'visual' in layer:
                    parms.grad *= coeff_v
            elif args.dataset in VV_Datasets:
                if 'visual_net1' in layer:
                    parms.grad *= coeff_a
                if 'visual_net2' in layer:
                    parms.grad *= coeff_v
            else:
                raise NotImplementedError

def QMF_prediciton(out_a, out_v):
    audio_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
    visual_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))

    audio_conf = audio_energy / 10
    visual_conf = visual_energy / 10
    audio_conf = torch.reshape(audio_conf, (-1, 1))
    visual_conf = torch.reshape(visual_conf, (-1, 1))
    out = (out_a * audio_conf.detach() + out_v * visual_conf.detach())
    return out


def OGM_modulation_AVT(args, model, out_a, out_v, out_t, label, epoch, logger):
    score_a = sum([F.softmax(out_a, dim=1)[i][label[i]] for i in range(out_a.size(0))])
    score_v = sum([F.softmax(out_v, dim=1)[i][label[i]] for i in range(out_v.size(0))])
    score_t = sum([F.softmax(out_t, dim=1)[i][label[i]] for i in range(out_v.size(0))])

    min_score = min([score_a, score_v, score_t])
    ratio_a = score_a / min_score
    ratio_v = score_v / min_score
    ratio_t = score_t / min_score
    
    logger.statistics_add('OGM/ratio_a', ratio_a)
    logger.statistics_add('OGM/ratio_v', ratio_v)
    logger.statistics_add('OGM/ratio_t', ratio_t)
    
    if args.modulation_starts <= epoch <= args.modulation_ends:
        coeff_a = 1 - F.tanh(args.OGM_alpha * F.relu(ratio_a)) if ratio_a > 1 else 1
        coeff_v = 1 - F.tanh(args.OGM_alpha * F.relu(ratio_v)) if ratio_v > 1 else 1
        coeff_t = 1 - F.tanh(args.OGM_alpha * F.relu(ratio_t)) if ratio_t > 1 else 1
        
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[1]

            if 'audio' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':
                    parms.grad = parms.grad * coeff_a + \
                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                        parms.grad *= coeff_a

            if 'visual' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':
                    parms.grad = parms.grad * coeff_v + \
                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_v
            
            if 'text' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':
                    parms.grad = parms.grad * coeff_t + \
                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_t


def MLA_modulation_AVT(args, model, criterion, optimizer, a, v, t, label, batch_step, len_dataloader):
    out_a = model.module.fusion_module.forward_a(a)
    loss_a = criterion(out_a, label)
    loss_a.backward()
    model.module.mla_gs_plugin.before_update(model.module.fusion_module.fc_out, a, 
                                    batch_step, len_dataloader,  model.module.mla_gs_plugin.exp_count)
    optimizer.step()
    optimizer.zero_grad()
    model.module.mla_gs_plugin.exp_count += 1
            
    out_v = model.module.fusion_module.forward_v(v)
    loss_v = criterion(out_v, label)
    loss_v.backward()
    model.module.mla_gs_plugin.before_update(model.module.fusion_module.fc_out, v, 
                            batch_step, len_dataloader,  model.module.mla_gs_plugin.exp_count)
    optimizer.step()
    optimizer.zero_grad()
    model.module.mla_gs_plugin.exp_count += 1
    
    out_t = model.module.fusion_module.forward_t(t)
    loss_t = criterion(out_t, label)
    loss_t.backward()
    model.module.mla_gs_plugin.before_update(model.module.fusion_module.fc_out, t, 
                            batch_step, len_dataloader,  model.module.mla_gs_plugin.exp_count)
    optimizer.step()
    optimizer.zero_grad()
    model.module.mla_gs_plugin.exp_count += 1
    
    return loss_a+loss_v+loss_t, loss_a, loss_v, loss_t


def MLA_prediction_AVT(out_a, out_v, out_t):
    out_list = torch.cat([out_a[None,:], out_v[None,:], out_t[None,:]])
    softmax_n1 = nn.Softmax(dim=-1)
    softmax_0 = nn.Softmax(dim=0)
    predict = softmax_n1(out_list)
    entropy = - predict * torch.log(predict) 
    entropy = entropy.sum(dim=-1, keepdims=True) 
    lambda_mr = - entropy
    out = softmax_0(lambda_mr) * out_list
    return out.sum(dim=0)


def AGM_modulation_AVT(args, model, out_a, out_v, out_t, label, epoch, logger):
    if torch.isnan(out_a).any() or torch.isnan(out_v).any():
        raise ValueError
    
    idx = torch.arange(0, label.shape[0]).to(label.device)
    score_audio = torch.clamp(F.softmax(out_a, dim=-1), min=1e-8)
    score_audio = -torch.log(score_audio[idx, label]).mean()
    score_visual = torch.clamp(F.softmax(out_v, dim=-1), min=1e-8)
    score_visual = -torch.log(score_visual[idx, label]).mean()
    score_text = torch.clamp(F.softmax(out_t, dim=-1), min=1e-8)
    score_text = -torch.log(score_text[idx, label]).mean()
    
    min_score = min([score_audio, score_visual, score_text])
    ratio_a = math.exp(min_score.item() - score_audio.item())
    ratio_v = math.exp(min_score.item() - score_visual.item())
    ratio_t = math.exp(min_score.item() - score_text.item())
    logger.statistics_add('AGM/ratio_a', ratio_a)
    logger.statistics_add('AGM/ratio_v', ratio_v)
    logger.statistics_add('AGM/ratio_t', ratio_t)
    
    if args.modulation_starts <= epoch <= args.modulation_ends:
        train_score_a = model.module.AGM_module.train_score_a
        train_score_v = model.module.AGM_module.train_score_v
        train_score_t = model.module.AGM_module.train_score_t
        
        min_train_score = min([train_score_a, train_score_v, train_score_t])
        optimal_ratio_a = math.exp(min_train_score - train_score_a)
        optimal_ratio_v = math.exp(min_train_score - train_score_v)
        optimal_ratio_t = math.exp(min_train_score - train_score_t)
        
        coeff_a = math.exp(args.AGM_alpha*(min(optimal_ratio_a - ratio_a,10)))
        coeff_v = math.exp(args.AGM_alpha*(min(optimal_ratio_v - ratio_v,10)))
        coeff_t = math.exp(args.AGM_alpha*(min(optimal_ratio_t - ratio_t,10)))
        
        model.module.AGM_module.updata_train_score(score_audio, score_visual, score_text)
        
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[1]
            if 'audio' in layer:
                parms.grad *= coeff_a
            if 'visual' in layer:
                parms.grad *= coeff_v
            if 'text' in layer:
                parms.grad *= coeff_t


def QMF_prediciton_AVT(out_a, out_v, out_t):
    audio_energy = torch.log(torch.sum(torch.exp(out_a), dim=1))
    visual_energy = torch.log(torch.sum(torch.exp(out_v), dim=1))
    text_energy = torch.log(torch.sum(torch.exp(out_t), dim=1))

    audio_conf = audio_energy / 10
    visual_conf = visual_energy / 10
    text_conf = text_energy / 10
    
    audio_conf = torch.reshape(audio_conf, (-1, 1))
    visual_conf = torch.reshape(visual_conf, (-1, 1))
    text_conf = torch.reshape(text_conf, (-1, 1))
    
    out = (out_a * audio_conf.detach() + out_v * visual_conf.detach() + out_t * text_conf.detach())
    return out


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( np.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = np.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, np.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        new_dps={}
        new_init_sol=[]

        for item in dps:
            new_dps[item]=dps[item].numpy()

        for item in init_sol:
            if(torch.is_tensor(item)):
                data=item.numpy()
            else:
                data=item
            new_init_sol.append(data)
        
        dps=new_dps
        init_sol=new_init_sol


        
        n=len(vecs)
        sol_vec = np.zeros(n)

        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

def get_record_names_audio_and_record_names_visual(args, model):
    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'fusion_module' in name: 
            continue
        
        if args.dataset in AV_Datasets:
            if ('audio' in name):
                record_names_audio.append((name, param))
                continue
            if ('visual' in name):
                record_names_visual.append((name, param))
                continue
        elif args.dataset in VV_Datasets:
            if ('visual_net1' in name):
                record_names_audio.append((name, param))
                continue
            if ('visual_net2' in name):
                record_names_visual.append((name, param))
                continue
    return record_names_audio, record_names_visual
    

def MMPareto(args, model, out, out_a, out_v, label, optimizer):
    criterion = nn.CrossEntropyLoss()
    loss_mm = criterion(out, label)
    loss_a=criterion(out_a,label)
    loss_v=criterion(out_v,label)
    
    record_names_audio, record_names_visual = get_record_names_audio_and_record_names_visual(args, model)

    losses=[loss_mm,loss_a,loss_v]
    all_loss = ['both', 'audio', 'visual']

    grads_audio = {}
    grads_visual={}


    for idx, loss_type in enumerate(all_loss):
        loss = losses[idx]
        loss.backward(retain_graph=True)

        if(loss_type=='visual'):
            for tensor_name, param in record_names_visual:
                if not param.requires_grad:
                    continue
                if loss_type not in grads_visual.keys():
                    grads_visual[loss_type] = {}
                grads_visual[loss_type][tensor_name] = param.grad.data.clone()
            grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])

        elif(loss_type=='audio'):
            for tensor_name, param in record_names_audio:
                if not param.requires_grad:
                    continue
                if loss_type not in grads_audio.keys():
                    grads_audio[loss_type] = {}
                grads_audio[loss_type][tensor_name] = param.grad.data.clone()
            grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])

        else:
            for tensor_name, param in record_names_audio:
                if not param.requires_grad:
                    continue
                if loss_type not in grads_audio.keys():
                    grads_audio[loss_type] = {}
                grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
            grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
            for tensor_name, param in record_names_visual:
                if not param.requires_grad:
                    continue
                if loss_type not in grads_visual.keys():
                    grads_visual[loss_type] = {}
                grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
            grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

        optimizer.zero_grad()
    
    this_cos_audio=F.cosine_similarity(grads_audio['both']["concat"],grads_audio['audio']["concat"],dim=0)
    this_cos_visual=F.cosine_similarity(grads_visual['both']["concat"],grads_visual['visual']["concat"],dim=0)

    audio_task=['both','audio']
    visual_task=['both','visual']

    # audio_k[0]: weight of multimodal loss
    # audio_k[1]: weight of audio loss
    # if cos angle <0 , solve pareto
    # else use equal weight

    audio_k=[0,0]
    visual_k=[0,0]

    if(this_cos_audio>0):
        audio_k[0]=0.5
        audio_k[1]=0.5
    else:
        audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
    if(this_cos_visual>0):
        visual_k[0]=0.5
        visual_k[1]=0.5
    else:
        visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])

    gamma=1.5

    loss=loss_mm+loss_a+loss_v
    loss.backward()


    for name, param in model.named_parameters():
        if param.grad is not None:
            layer = re.split('[_.]',str(name))
            if('fusion_module' in layer):
                continue
            
            if args.dataset in AV_Datasets:
                if('audio' in layer):
                    three_norm=torch.norm(param.grad.data.clone())
                    new_grad=2*audio_k[0]*grads_audio['both'][name]+2*audio_k[1]*grads_audio['audio'][name]
                    new_norm=torch.norm(new_grad)
                    diff=three_norm/new_norm
                    if(diff>1):
                        param.grad=diff*new_grad*gamma
                    else:
                        param.grad=new_grad*gamma

                if('visual' in layer):
                    three_norm=torch.norm(param.grad.data.clone())
                    new_grad=2*visual_k[0]*grads_visual['both'][name]+2*visual_k[1]*grads_visual['visual'][name]
                    new_norm=torch.norm(new_grad)
                    diff=three_norm/new_norm
                    if(diff>1):
                        param.grad=diff*new_grad*gamma
                    else:
                        param.grad=new_grad*gamma
            elif args.dataset in VV_Datasets:
                if('net1' in layer):
                    three_norm=torch.norm(param.grad.data.clone())
                    new_grad=2*audio_k[0]*grads_audio['both'][name]+2*audio_k[1]*grads_audio['audio'][name]
                    new_norm=torch.norm(new_grad)
                    diff=three_norm/new_norm
                    if(diff>1):
                        param.grad=diff*new_grad*gamma
                    else:
                        param.grad=new_grad*gamma

                if('net2' in layer):
                    three_norm=torch.norm(param.grad.data.clone())
                    new_grad=2*visual_k[0]*grads_visual['both'][name]+2*visual_k[1]*grads_visual['visual'][name]
                    new_norm=torch.norm(new_grad)
                    diff=three_norm/new_norm
                    if(diff>1):
                        param.grad=diff*new_grad*gamma
                    else:
                        param.grad=new_grad*gamma

    return loss, loss_a, loss_v


def compute_alpha(grad1, grad2):
    numerator = torch.dot((grad2 - grad1), grad2)
    diff = grad1 - grad2
    denominator = torch.norm(diff, p=2) ** 2
    
    epsilon = 1e-8 
    alpha = numerator / (denominator + epsilon)
    
    alpha_hat = torch.clamp(alpha, min=0.0, max=1.0)
    return alpha_hat

def Pareto_TCMax(args, model, out, a, v, label, optimizer):
    from .losses import TCMax_loss
    TCMax, loss_a,  loss_v = TCMax_loss(args, model, a, v, label)
    criterion = nn.CrossEntropyLoss()
    loss_mm = criterion(out, label)
    
    losses=[loss_mm, TCMax]
    all_loss = ['Joint', 'TCMax']
    
    grads = {}
    
    # Get all grads
    for idx, loss_type in enumerate(all_loss):
        loss = losses[idx]
        loss.backward(retain_graph=True)
        for tensor_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if loss_type not in grads.keys():
                grads[loss_type] = {}
            grads[loss_type][tensor_name] = param.grad.data.clone()
        grads[loss_type]["concat"] = torch.cat([grads[loss_type][tensor_name].flatten()  for tensor_name in grads[loss_type].keys()])
        
        optimizer.zero_grad()
    
    cos_Joint_TCMax = F.cosine_similarity(grads['Joint']["concat"], grads['TCMax']["concat"],dim=0)
    if(cos_Joint_TCMax>0):
        # No grad conflict
        alpha = 0.5
    else:
        # With grad conflict
        alpha = compute_alpha(grads['Joint']["concat"], grads['TCMax']["concat"])
    
    loss = TCMax
    loss.backward()
    
    for tensor_name, param in model.named_parameters():
        if param.grad is not None:
            if tensor_name in grads['Joint'].keys():
                merge_grad_norm=torch.norm(param.grad.data.clone())
                new_grad = alpha * grads['Joint'][tensor_name] + (1-alpha) * grads['TCMax'][tensor_name]
                new_norm=torch.norm(new_grad)
                diff=merge_grad_norm/new_norm
                
                # 初期，未修改
                param.grad = new_grad * 2.0
                
    return loss, TCMax, loss_a, loss_v, alpha