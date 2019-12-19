import torch

def prepare_optim(model, opt):
    """
    prepare the optimizer from the trainable parameters from the model.
    args:
        model: the neural decision forest to be trained
        opt: experiment configuration object
    """
    params = [ p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-5)
    # For CIFAR-10, use a scheduler to shrink learning rate
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[150, 250], 
                                                     gamma=0.3)

    return optimizer, scheduler