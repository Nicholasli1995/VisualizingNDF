import torch
import argparse
import utils
from dataset import prepare_db

def parse_arg():
    parser = argparse.ArgumentParser(description='ndf_vis.py')
    parser.add_argument('-dataset', choices=['mnist','cifar10'], default='cifar10')
    parser.add_argument('-show_data', type=bool, default=False)
    parser.add_argument('-gpuid', type=int, default=0)
    return parser.parse_args()

# configuration
opt = parse_arg()

torch.cuda.set_device(opt.gpuid)

# This script visualize and help understanding deep neural decision forest.
if opt.dataset == 'mnist':
    model_path = "../pre-trained/mnist_depth_9_tree_1_acc_0.993.pth"
elif opt.dataset == 'cifar10':
    model_path = "../pre-trained/cifar10_depth_9_tree_1_ResNet50_acc_0.9341.pth"
else:
    raise NotImplementedError

# load model
model = torch.load(model_path).cuda()

# prepare dataset
db = prepare_db(opt)
dataset = db['eval']
if opt.show_data:
    utils.show_data(dataset, opt.dataset)
    
# compute saliency map
# pick a tree index and splitting node index
tree_idx = 0
node_idx = 0 # 0 - 510 for 511 splitting nodes

# get saliency maps for a specified node for different input tensors
# utils.get_node_saliency_map(dataset, model, tree_idx, node_idx, name=opt.dataset)

# get the computational paths for the input
sample, paths, class_pred = utils.get_paths(dataset, model, tree_idx, name=opt.dataset)

# for each sample, plot the saliency and how the input will influence the 
# decision making
utils.get_path_saliency(sample, paths, class_pred, model, tree_idx, name=opt.dataset)
  