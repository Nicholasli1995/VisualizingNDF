import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='parse.py')
    # choice of dataset
    parser.add_argument('-dataset', choices=['mnist','cifar10'], 
                        default='cifar10')
    # batch size used when training the feature extractor
    parser.add_argument('-batch_size', type=int, default = 256)
    
    parser.add_argument('-feat_dropout', type=float, default = 0)
    
    # how many tree to use
    parser.add_argument('-n_tree', type=int, default=1)
    # tree depth
    parser.add_argument('-tree_depth', type=int, default=9)
    # number of classes for the dataset
    parser.add_argument('-n_class', type=int, default=10)
    # 
    parser.add_argument('-tree_feature_rate', type=float, default = 1)
    # learning rate
    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    # choice of GPU
    parser.add_argument('-gpuid', type=int, default=0)
    # total number of training epochs
    parser.add_argument('-epochs', type=int, default=350)
    # log every how many batches
    parser.add_argument('-report_every', type=int, default=20)
    # whether to save the trained model
    parser.add_argument('-save', type=bool, default=True)
    # path to save the trained model
    parser.add_argument('-save_dir', type=str, default='../pre-trained')
    # network architecture to use
    parser.add_argument('-model_type', type=str, default='resnet18')    
    #
    parser.add_argument('-init_feat_map_num', type=int, default=64)
    # batch size used when update the leaf node prediction vectors
    parser.add_argument('-label_batch_size', type=int, default= 2240)
    # representation length after the last FC layer
    parser.add_argument('-feature_length', type=int, default=1024)
    # number of threads for data loading
    parser.add_argument('-num_worker', type=int, default=4)    
    opt = parser.parse_args()
    return opt