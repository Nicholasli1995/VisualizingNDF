import logging
import os
import torchvision
import torchvision.transforms as transforms

def prepare_db(opt):
    """
    prepare the Pytorch dataset object for classification. 
    args:
        opt: the experiment configuration object.
    return:
        a dictionary contraining the training and evaluation dataset
    """
    logging.info("Use %s dataset"%(opt.dataset))
    
    # prepare MNIST dataset
    if opt.dataset == 'mnist':
        opt.n_class = 10     
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
    # prepare CIFAR-10 dataset
    elif opt.dataset == 'cifar10':
        opt.n_class = 10 
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
    elif opt.dataset == 'Nexperia':
        # Nexperia image classification
        # Updated 2019/12/01
        # Reference: https://www.kaggle.com/c/semi-conductor-image-classification-first
        # Task: classify whether a semiconductor image is abnormal
        opt.n_class = 2     
        if not os.path.exists("../data/Nexperia/semi-conductor-image-classification-first"):
            logging.info("Data not found. Please download first.")
            raise ValueError
        import sys
        sys.path.insert(0, "../data/")
        from Nexperia.dataset import get_datasets
        return get_datasets(opt)
    else:
        raise NotImplementedError
    return {'train':train_dataset, 'eval':eval_dataset}
    