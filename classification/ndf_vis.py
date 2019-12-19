"""
Visualizing the decision saliency maps for pre-trained deep neural decison forest models. 
"""
import utils
from dataset import prepare_db

import torch
import argparse

def parse_arg():
    """
    argument parser.
    """
    parser = argparse.ArgumentParser(description='ndf_vis.py')
    # which dataset to use, mnist or cifar10
    parser.add_argument('-dataset', choices=['mnist', 'cifar10', 'Nexperia'], default='cifar10')
    # which GPU to use
    parser.add_argument('-gpuid', type=int, default=0)
    # root path for Nexperia dataset
    parser.add_argument('-nexperia_root', type=str, default='../data/Nexperia/semi-conductor-image-classification-first')    
    # use all images from Nexperia dataset
    parser.add_argument('-train_all', type=bool, default=False)    
    # if not, specify the used fraction for Nexperia dataset
    parser.add_argument('-train_ratio', type=float, default=0.9) 
    return parser.parse_args()

# parse arguments
opt = parse_arg()

# For now only GPU version is supported
torch.cuda.set_device(opt.gpuid)

# please place the downloaded pre-trained models in the following directory
if opt.dataset == 'mnist':
    model_path = "../pre-trained/mnist_depth_9_tree_1_acc_0.993.pth"
elif opt.dataset == 'cifar10':
    model_path = "../pre-trained/cifar10_depth_9_tree_1_ResNet50_acc_0.9341.pth"
elif opt.dataset == 'Nexperia':
    model_path = "../pre-trained/nexperia_depth_9_tree_1_resnet50_acc_0.955.pth"
else:
    raise NotImplementedError

# load model
model = torch.load(model_path).cuda()

# prepare dataset
db = prepare_db(opt)
# use only the evaluation subset. use db['train'] for fetching the training subset
dataset = db['eval']
 
# ==================================================================================
# compute saliency maps for different inputs for one splitting node
# pick a tree index and splitting node index
# tree_idx = 0
# node_idx = 0 # 0 - 510 for the 511 splitting nodes in a tree of depth 9
# get saliency maps for a specified node for different input tensors
# utils.get_node_saliency_map(dataset, model, tree_idx, node_idx, name=opt.dataset)
# ==================================================================================

# visualize for the first tree (modify to others if needed)
tree_idx = 0

# get the computational paths for the some random inputs
sample, paths, class_pred = utils.get_paths(dataset, model, tree_idx, name=opt.dataset)

# for each sample, compute and plot the decision saliency map, which reflects how the input will influence the 
# decision-making process
utils.get_path_saliency(sample, paths, class_pred, model, tree_idx, name=opt.dataset)
  
