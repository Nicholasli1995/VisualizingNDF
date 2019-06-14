import logging
import torchvision
import torchvision.transforms as transforms

def prepare_db(opt):
    """
    prepare the used datasets. 
    args:
        opt: the experiment configuration object.
    """
    logging.info("Use %s dataset"%(opt.dataset))
    
    # prepare MNIST dataset
    if opt.dataset == 'mnist':
        # training set
        train_dataset = torchvision.datasets.MNIST('../data/mnist', train=True, 
                                                   download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ]))

        # evaluation set
        eval_dataset = torchvision.datasets.MNIST('../data/mnist', train=False, 
                                                  download=True,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), 
                                                                            (0.3081,))
                                                   ]))
        return {'train':train_dataset,'eval':eval_dataset}

    # prepare CIFAR-10 dataset
    elif opt.dataset == 'cifar10':
        # define the image transformation operators
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])        
        train_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', 
                                                     train=True, 
                                                     download=True, 
                                                     transform=transform_train)
        eval_dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', 
                                                    train=False, 
                                                    download=True, 
                                                    transform=transform_test)
        return {'train':train_dataset,'eval':eval_dataset}

    else:
        raise NotImplementedError