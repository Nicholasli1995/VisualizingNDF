"""
Visualize Deep Neural Deicsion Forest For Facial Age Estimaiton
@author: Shichao (Nicholas) Li
Contact: nicholas.li@connect.ust.hk
License: MIT
"""
import torch
import argparse
import vis_utils
from data_prepare import prepare_db
# This script visualize and help understanding deep neural decision forest.

def parse_arg():
    parser = argparse.ArgumentParser(description='ndf_vis.py')
    parser.add_argument('-gpuid', type=int, default=0)    
    parser.add_argument('-dataset_name', type=str, default='CACD')  
    parser.add_argument('-image_size', type=int, default=256)  
    parser.add_argument('-crop_size', type=int, default=224)  
    # whether to create cache for the images to avoid reading disks
    parser.add_argument('-cache', type=bool, default=False)    
    # whether to apply data augmentation by applying random transformation
    parser.add_argument('-transform', type=bool, default=True)
    # whether to apply data augmentation by multiple shape initialization
    parser.add_argument('-augment', type=bool, default=False)
    # how many times to create different initializations for every image
    # whether to use the training set of CACD dataset for training
    parser.add_argument('-cacd_train', type=bool, default=True)
    # whether to plot images after dataset initialization
    parser.add_argument('-visualize', type=bool, default=False)
    parser.add_argument('-gray_scale', type=bool, default=False)    
    return parser.parse_args()

# get configuration
opt = parse_arg()

# use GPU
torch.cuda.set_device(opt.gpuid)

if opt.dataset_name == 'Morph':
    # Sorry that the MORPH dataset is currently not freely available. 
    # For now I can not release my pre-processed dataset without permission.
    raise ValueError
elif opt.dataset_name == 'CACD':
    model_path = "../../pre-trained/CACD_MAE_4.59.pth"
else:
    raise NotImplementedError

# load model
model = torch.load(model_path)
model.cuda()

# prepare dataset
db = prepare_db(opt)
dataset = db['eval'][0]
    
# compute saliency map
# pick a tree within the forest 
tree_idx = 0
depth = model.forest.trees[0].depth
# pick a splitting node index (optional)
#node_idx = 0 # 0 - 510 for 511 splitting nodes 

# get saliency maps for a specified node for different input tensors
# vis_utils.get_node_saliency_map(dataset, model, tree_idx, node_idx, name=opt.dataset)

# get the computational paths for the input
sample, labels, paths, class_pred = vis_utils.get_paths(dataset, model, 
                                                        name=opt.dataset_name,
                                                        depth=depth)

# for each sample, plot the saliency and visualize how the input influence the 
# decision making process
vis_utils.get_path_saliency(sample, labels, paths, class_pred, model, tree_idx, 
                            name=opt.dataset_name)
