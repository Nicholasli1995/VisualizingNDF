"""
Optimizer preparation
"""
import torch

def prepare_optim(model, opt):
    params = [ p for p in model.parameters() if p.requires_grad]
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params, lr = opt.lr, 
                                     weight_decay = opt.weight_decay)
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr = opt.lr, 
                                    momentum = opt.momentum,
                                    weight_decay = opt.weight_decay)    
    # scheduler with pre-defined learning rate decay
#    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                     milestones = opt.milestones, 
#                                                     gamma = opt.gamma)
    # automatically decrease learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=10,
                                                           verbose=True,
                                                           min_lr=0.01)
    return optimizer, scheduler