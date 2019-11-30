"""
Initialize a RNDF.
"""
import ndf

def prepare_model(opt):
    # RNDF consists of two parts:
    #1. a feature extraction model using residual learning
    #2. a neural decision forst
    feat_layer = ndf.FeatureLayer(model_type = opt.model_type, 
                                  num_output = opt.num_output, 
                                  gray_scale = opt.gray_scale,
                                  input_size = opt.image_size,
                                  pretrained = opt.pretrained)    
    forest = ndf.Forest(opt.n_tree, opt.tree_depth, opt.num_output, 
                        1, opt.cuda)
    model = ndf.NeuralDecisionForest(feat_layer, forest)     
    if opt.cuda:
        model = model.cuda()
    else:
        raise NotImplementedError

    return model