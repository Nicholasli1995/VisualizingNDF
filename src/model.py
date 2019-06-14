import ndf

def prepare_model(opt):
    """
    prepare the neural decison forest model. The model is composed of the 
    feature extractor (a CNN) and a decision forest. The feature extractor
    extracts features from input, which is sent to the decison forest for 
    inference.
    args:
        opt: experiment configuration object
    """
    # initialize feature extractor
    if opt.dataset == 'mnist':
        feat_layer = ndf.MNISTFeatureLayer(opt.feat_dropout, 
                                           opt.feature_length)
    elif opt.dataset == 'cifar10':
        feat_layer = ndf.CIFAR10FeatureLayer(opt.feat_dropout, 
                                             feat_length=opt.feature_length,
                                             archi_type=opt.model_type)
    else:
        raise NotImplementedError 
    # initialize the decison forest
    forest = ndf.Forest(n_tree = opt.n_tree, tree_depth = opt.tree_depth, 
                        feature_length = opt.feature_length,
                        vector_length = opt.n_class, use_cuda = opt.cuda)
    model = ndf.NeuralDecisionForest(feat_layer, forest)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model