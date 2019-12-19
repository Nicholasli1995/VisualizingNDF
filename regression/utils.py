"""
miscellaneous utility functions
"""
import argparse
import time
import logging

import numpy as np
from torch import save
from os import path as path
import os 

# function for parsing input arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='train.py')
    ## paths
    parser.add_argument('-img_path', type=str, default='../data/CACD/crop')
    parser.add_argument('-save_dir', type=str, default='../model/')
    parser.add_argument('-save_his_dir', type=str, default='../history/CACD')
    parser.add_argument('-test_model_path', type=str, default='../model/CACD_MAE_4.59.pth')
    ##-----------------------------------------------------------------------##
    ## model settings
    parser.add_argument('-save_name', type=str, default='trained_model') 
    parser.add_argument('-num_output', type=int, default=128) # only used for coupled routing functions      
    parser.add_argument('-model_type', type=str, default='hybrid') # only used for coupled routing functions (hierarchy = False)
    parser.add_argument('-n_tree', type=int, default=5)
    parser.add_argument('-tree_depth', type=int, default=6)
    parser.add_argument('-leaf_node_type', type=str, default='simple') 
    parser.add_argument('-pretrained', type=bool, default=False) # only used for coupled routing functions
    ## training settings
    # choose one tree for update at a time
    parser.add_argument('-dropout', type=bool, default=False) 
    parser.add_argument('-batch_size', type=int, default=30)
    # random seed 
    parser.add_argument('-seed', type=int, default=2019)
    # batch size used when updating label prediction
    parser.add_argument('-label_batch_size', type=int, default=500)
    # number of threads to use when loading data
    parser.add_argument('-num_threads', type=int, default=4)
    # update leaf node distribution every certain number of network training
    parser.add_argument('-update_every', type=int, default=50)
    # how many iterations to update prediction in each leaf node 
    parser.add_argument('-label_iter_time', type=int, default=20)
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-report_every', type=int, default=40)
    # whether to perform evaluation on evaluation set during training
    parser.add_argument('-eval', type=bool, default=True)
    # whether to record and report loss history at the end of training
    parser.add_argument('-history', type=bool, default=False)  
    parser.add_argument('-eval_every', type=int, default=100)
    # threshold using for computing CS
    parser.add_argument('-threshold', type=int, default=5)
    ##-----------------------------------------------------------------------##    
    ## dataset settings
    parser.add_argument('-dataset_name', type=str, default='CACD')  
    parser.add_argument('-image_size', type=int, default=256)  
    parser.add_argument('-crop_size', type=int, default=224)  
    # whether to create cache for the images to avoid reading disks
    parser.add_argument('-cache', type=bool, default=False)    
    # whether to apply data augmentation by applying random transformation
    parser.add_argument('-transform', type=bool, default=True)
    # whether to apply data augmentation by multiple shape initialization
    parser.add_argument('-augment', type=bool, default=False)
    # whether to use the training set of CACD dataset for training
    parser.add_argument('-cacd_train', type=bool, default=True)
    # whether to plot images after dataset initialization
    parser.add_argument('-gray_scale', type=bool, default=False)
    ##-----------------------------------------------------------------------##    
    # Optimizer settings
    parser.add_argument('-optim_type', type=str, default='sgd')
    parser.add_argument('-lr', type=float, default=0.5, help="sgd: 0.5, adam: 0.001")
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-momentum', type=float, default=0.9, help="sgd: 0.9")
    # reduce the learning rate after each milestone
    #parser.add_argument('-milestones', type=list, default=[6, 12, 18])
    parser.add_argument('-milestones', type=list, default=[2,4,6,8])
    # how much to reduce the learning rate
    parser.add_argument('-gamma', type=float, default=0.5)
    ##-----------------------------------------------------------------------##  
    ## usage configuration
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-evaluate', type=bool, default=False)
    opt = parser.parse_args()
    return opt

def get_save_dir(opt, str_type=None):
    if str_type == 'his':
        root = opt.save_his_dir
    else:
        root = opt.save_dir
    save_name = path.join(root, opt.save_name) 
    save_name += '_model_type_' 
    save_name += opt.model_type
    save_name += '_RNDF_'
    save_name += '_depth{:d}_tree{:d}_output{:d}'.format(opt.tree_depth, opt.n_tree, opt.num_output)
    save_name += time.asctime(time.localtime(time.time()))
    save_name += '.pth'
    return save_name

def save_model(model, opt):
    save_name = get_save_dir(opt)
    save(model, save_name)
    return

def save_best_model(model, path):
    save(model, path)
    return

def update_log(best_model_dir, MAE, CS):
    text = time.asctime(time.localtime(time.time())) + ' '
    text += "Current MAE: " + MAE[0] + " Current CS: " + CS + " "
    text += "Best MAE: " + MAE[1] + "\r\n"
    with open(os.path.join(best_model_dir, "log.txt"), "a") as myfile:
        myfile.write(text)
    return

def save_history(train_his, eval_his, opt):
    save_name = get_save_dir(opt, 'his')
    train_his_name = save_name +'train_his_stage' 
    eval_his_name = save_name + 'eval_his_stage' 
    if not path.exists(opt.save_his_dir):
        os.mkdir(opt.save_his_dir)
    save(train_his, train_his_name)
    save(eval_his, eval_his_name)
    
def split_dic(data_dic):
    img_path_list = []
    age_list = []
    for key in data_dic:
        img_path_list += data_dic[key]['path']
        age_list += data_dic[key]['age_list']
    img_path_list = np.array(img_path_list)
    age_list = np.array(age_list)
    total_imgs = len(img_path_list)
    random_indices = np.random.choice(total_imgs, total_imgs, replace=False)
    num_train = int(len(img_path_list)*0.8)
    train_path_list = list(img_path_list[random_indices[:num_train]])
    train_age_list = list(age_list[random_indices[:num_train]])
    valid_path_list = list(img_path_list[random_indices[num_train:]])
    valid_age_list = list(age_list[random_indices[num_train:]])
    train_dic = {'path':train_path_list, 'age_list':train_age_list}
    valid_dic = {'path':valid_path_list, 'age_list':valid_age_list}
    return train_dic, valid_dic

def check_split(train_dic, eval_dic):
    train_num = len(train_dic['path'])
    valid_num = len(eval_dic['path'])
    logging.info("Image split: {:d} training, {:d} validation".format(train_num, valid_num))
    logging.info("Total unique image num: {:d} ".format(len(set(train_dic['path'] + eval_dic['path']))))
    return