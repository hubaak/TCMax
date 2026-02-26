import torch
import torch.nn as nn


def Naive_fusion(args, model, criterion, optimizer, data_packet, device, params):
    spec, image, label, idx = data_packet
    spec = spec.to(device)
    image = image.to(device)
    label = label.to(device)
    a, v, out = model(spec.unsqueeze(1).float(), image.float())
    out_a = model.module.fusion_module.forward_a(a)
    out_v = model.module.fusion_module.forward_v(v)
    
    loss = criterion(out, label)
    if out_a is not None and out_v is not None:
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)
    else:
        loss_a = loss_v = -1
    
    loss.backward()
    optimizer.step()
    
    log = {}
    log['loss'] = loss.item()
    log['loss_a'] = loss_a.item()
    log['loss_v'] = loss_v.item()
    return log

def OGM_GE(args, model, criterion, optimizer, data_packet, device, params):
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    
    spec, image, label, idx = data_packet
    spec = spec.to(device)
    image = image.to(device)
    label = label.to(device)
    a, v, out = model(spec.unsqueeze(1).float(), image.float())
    out_a = model.module.fusion_module.forward_a(a)
    out_v = model.module.fusion_module.forward_v(v)
    
    loss = criterion(out, label)
    if out_a is not None and out_v is not None:
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)
    else:
        loss_a = loss_v = -1
    
    
    # Modulation starts here !
    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v

    """
     Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
    """

    if ratio_v > 1:
        coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
        coeff_a = 1
    else:
        coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
        coeff_v = 1

    if args.use_tensorboard:
        iteration = params['epoch'] * len(params['dataloader']) + params['step']
        params['writer'].add_scalar('data/ratio v', ratio_v, iteration)
        params['writer'].add_scalar('data/coefficient v', coeff_v, iteration)
        params['writer'].add_scalar('data/coefficient a', coeff_a, iteration)

    if args.modulation_starts <= params['epoch'] <= args.modulation_ends: # bug fixed
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[1]

            if 'audio' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':  # bug fixed
                    parms.grad = parms.grad * coeff_a + \
                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_a

            if 'visual' in layer and len(parms.grad.size()) == 4:
                if args.modulation == 'OGM_GE':  # bug fixed
                    parms.grad = parms.grad * coeff_v + \
                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    parms.grad *= coeff_v
    else:
        pass
    
    loss.backward()