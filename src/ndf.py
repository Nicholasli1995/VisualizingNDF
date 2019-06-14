import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import resnet
# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)
FLT_MAX = float(np.finfo(np.float32).max)

class MNISTFeatureLayer(nn.Module):
    def __init__(self, dropout_rate, feat_length = 512, shallow=False):
        super(MNISTFeatureLayer, self).__init__()
        self.shallow = shallow
        self.feat_length = feat_length
        if shallow:
            self.add_module('conv1', nn.Conv2d(1, 32, kernel_size=15,padding=1,stride=5))
        else:
            self.conv_layers = nn.Sequential()
            self.conv_layers.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
            self.conv_layers.add_module('bn1', nn.BatchNorm2d(32))
            self.conv_layers.add_module('relu1', nn.ReLU())
            self.conv_layers.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop1', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.conv_layers.add_module('bn2', nn.BatchNorm2d(64))
            self.conv_layers.add_module('relu2', nn.ReLU())
            self.conv_layers.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop2', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.conv_layers.add_module('bn3', nn.BatchNorm2d(128))
            self.conv_layers.add_module('relu3', nn.ReLU())
            self.conv_layers.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop3', nn.Dropout(dropout_rate))
        self.linear_layer = nn.Linear(self.get_conv_size(), 
                                      feat_length,
                                      bias=True)
    def get_out_feature_size(self):
        return self.feat_length

    def get_conv_size(self):
        if self.shallow:
            return 64*4*4
        else:
            return 128*3*3
        
    def forward(self, x):
        feats = self.conv_layers(x)
        feats = feats.view(x.size()[0], -1)
        return self.linear_layer(feats)
    
class CIFAR10FeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, feat_length = 512, archi_type='resnet18'):
        super(CIFAR10FeatureLayer, self).__init__()
        self.archi_type = archi_type
        self.feat_length = feat_length
        if self.archi_type == 'default':
            self.add_module('conv1', nn.Conv2d(3, 32, kernel_size=3, padding=1))
            self.add_module('bn1', nn.BatchNorm2d(32))
            self.add_module('relu1', nn.ReLU())
            self.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop1', nn.Dropout(dropout_rate))
            self.add_module('conv2', nn.Conv2d(32, 32, kernel_size=3, padding=1))
            self.add_module('bn2', nn.BatchNorm2d(32))
            self.add_module('relu2', nn.ReLU())
            self.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop2', nn.Dropout(dropout_rate))
            self.add_module('conv3', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.add_module('bn3', nn.BatchNorm2d(64))
            self.add_module('relu3', nn.ReLU())
            self.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop3', nn.Dropout(dropout_rate))
        elif self.archi_type == 'resnet18':
            self.add_module('resnet18', resnet.ResNet18(feat_length))
        elif self.archi_type == 'resnet50':
            self.add_module('resnet50', resnet.ResNet50(feat_length))            
        elif self.archi_type == 'resnet152':
            self.add_module('resnet152', resnet.ResNet152(feat_length))  
        else:
            raise NotImplementedError
            
    def get_out_feature_size(self):
        if self.archi_type == 'default':
            return 64*4*4
        elif self.archi_type == 'resnet18':
            return 512
        elif self.archi_type == 'vgg16':
            return self.feat_length
        else:
            raise NotImplementedError

class Tree(nn.Module):
    def __init__(self, depth, feature_length, vector_length, use_cuda = False):
        """
        Args:
            depth (int): depth of the neural decision tree.
            feature_length (int): number of neurons in the last feature layer
            vector_length (int): length of the mean vector stored at each tree leaf node
        """
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        self.is_cuda = use_cuda
        # used in leaf node update 
        self.mu_cache = []
        
        onehot = np.eye(feature_length)
        # randomly use some neurons in the feature layer to compute decision function
        self.using_idx = np.random.choice(feature_length, self.n_leaf, replace=False)
        self.feature_mask = onehot[self.using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # a leaf node contains a mean vector and a covariance matrix
        self.pi = np.zeros((self.n_leaf, self.vector_length))    
        if not use_cuda:
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)
        else:
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor).cuda(), requires_grad=False)         
        # use sigmoid function as the decision function
        self.decision = nn.Sequential(OrderedDict([
                        ('sigmoid', nn.Sigmoid()),
                        ]))

    def forward(self, x, save_flag = False):
        """
        Args:
            param x (Tensor): input feature batch of size [batch_size,n_features]
        Return:
            (Tensor): routing probability of size [batch_size,n_leaf]
        """
#        def debug_hook(grad):
#            print('This is a debug hook')
#            print(grad.shape)
#            print(grad)        
        cache = {} # save some intermediate results for analysis
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask) # ->[batch_size,n_leaf]
        decision = self.decision(feats) # passed sigmoid->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision,dim=2) # ->[batch_size,n_leaf,1]
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]
        # for debug
        #decision.register_hook(debug_hook)
        # compute route probability
        # note: we do not use decision[:,0]
        # save some intermediate results for analysis
        if save_flag:
            cache['decision'] = decision[:,:,0]        
        batch_size = x.size()[0]
        
        mu = x.data.new(batch_size,1,1).fill_(1.)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**n_layer,2]
            mu = mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            mu = mu.view(batch_size, -1, 1)

        mu = mu.view(batch_size, -1)
        if save_flag:
            return mu, cache
        else:        
            return mu

    def pred(self, x):
        """
        Predict a vector based on stored vectors and routing probability
        Args:
            param x (Tensor): input feature batch of size [batch_size, feature_length]
        Return: 
            (Tensor): prediction [batch_size,vector_length]
        """
        p = torch.mm(self(x), self.pi)
        return p
    
    def get_pi(self):
        return self.pi
    
    def cal_prob(self, mu, pi):
        """

        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu,pi)
        return p
    
    def update_label_distribution(self, target_batches):
        """
        compute new mean vector based on a simple update rule inspired from traditional regression tree 
        Args:
            param feat_batch (Tensor): feature batch of size [batch_size, feature_length]
            param target_batch (Tensor): target batch of size [batch_size, vector_length]
        """
        with torch.no_grad():
            new_pi = self.pi.data.new(self.n_leaf, self.vector_length).fill_(0.) # Tensor [n_leaf,n_class] 
                
            for mu, target in zip(self.mu_cache, target_batches):
                prob = torch.mm(mu, self.pi)  # [batch_size,n_class]
                
                _target = target.unsqueeze(1) # [batch_size,1,n_class]
                _pi = self.pi.unsqueeze(0) # [1,n_leaf,n_class]
                _mu = mu.unsqueeze(2) # [batch_size,n_leaf,1]
                _prob = torch.clamp(prob.unsqueeze(1),min=1e-6,max=1.) # [batch_size,1,n_class]
    
                _new_pi = torch.mul(torch.mul(_target,_pi),_mu)/_prob # [batch_size,n_leaf,n_class]
                new_pi += torch.sum(_new_pi,dim=0)
        # test
        #import numpy as np
        #if np.any(np.isnan(new_pi.cpu().numpy())):
        #    print(new_pi)
        # test
        new_pi = F.softmax(new_pi, dim=1).data
      
        self.pi = Parameter(new_pi, requires_grad = False)
        return
    
class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, feature_length, vector_length, use_cuda = False):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = n_tree
        self.tree_depth = tree_depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        for _ in range(n_tree):
            tree = Tree(tree_depth, feature_length, vector_length, use_cuda)
            self.trees.append(tree)

    def forward(self, x, save_flag = False):
        predictions = []
        cache = []
        for tree in self.trees:
            if save_flag:
                mu, cache_tree = tree(x, save_flag = True)
                p = tree.cal_prob(mu, tree.get_pi())
                cache.append(cache_tree)
            else:    
                p = tree.pred(x)
            predictions.append(p.unsqueeze(2))
        prediction = torch.cat(predictions, dim=2)
        prediction = torch.sum(prediction, dim=2)/self.n_tree
        if save_flag:
            return prediction, cache
        else:
            return prediction

class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x, save_flag = False):
        feats = self.feature_layer(x)

        if save_flag:
            pred, cache = self.forest(feats, save_flag = True)
            return pred, cache, 0
        else:
            pred = self.forest(feats)
            return pred