from .vgg import VGG
from .paf_model import PAFModel
import torch.nn as nn
import torch


def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss(size_average = False)
    elif criterion == 'mse':
        return nn.MSELoss(size_average = False)
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def create_model(opt):
    if opt.model == 'vgg':
        backend = VGG()
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.model + ' not available.')
    model = PAFModel(backend,backend_feats,n_joints=19,n_paf=34,n_stages=7)
    if not opt.loadModel=='none':
        model = torch.load(opt.loadModel)
        print('Loaded model from '+opt.loadModel)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model, isdefault=Ture):
    if isdefault:
        return torch.optim.Adam(model.parameters(), opt.LR)
    else:
        params = get_parameters(opt, model)
        return torch.optim.Adam(params, opt.LR)
 
def get_parameters(opt, model)
    lr1 = []
    lr2 = []
    params_dict = dict(model.named_parameters())
    for key,value in params_dict.items():
        if value.requires_grad:
            if 'stage' in key:
                lr2.append(value)
            else:
                lr1.append(value)
    params = [{'params': lr1, 'lr': opt.LR},
              {'params': lr2, 'lr': opt.LR * 2}]              
    return params
