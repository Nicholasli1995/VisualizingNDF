"""
some utiliy functions for data processing and visualization.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch

# class name for CIFAR-10 dataset
cifar10_class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                      'frog', 'horse', 'ship', 'truck']

def show_data(dataset, name):
    """
    show some image from the dataset.
    args:
        dataset: dataset to show
        name: name of the dataset
    """
    if name == 'mnist':
        num_test = len(dataset)
        num_shown = 100
        cols = 10
        rows = int(num_shown/cols)
        indices = np.random.choice(list(range(num_test)), num_test)
        plt.figure()
        for i in range(num_shown):
            plt.subplot(rows, cols, i+1)
            plt.imshow(dataset[indices[i]][0].squeeze().numpy())
            plt.axis('off')
            plt.title(str(dataset[indices[i]][1].data.item()))
        plt.gcf().tight_layout()
        plt.show()
    else:
        raise NotImplementedError
    return

def get_sample(dataset, sample_num, name):
    # random seed
    #np.random.seed(2019)
    """
    get a batch of random images from the dataset
    args:
        dataset: Pytorch dataset object to use
        sample_num: number of samples to draw
        name: name of the dataset
    return:
        selected sample tensor
    """
    # get random indices
    indices = np.random.choice(list(range(len(dataset))), sample_num)
    if name in ['mnist', 'cifar10']:
        # for MNIST and CIFAR-10 dataset
        sample = [dataset[indices[i]][0].unsqueeze(0) for i in range(len(indices))]
        # concatenate the samples as one tensor
        sample = torch.cat(sample, dim = 0)
    else:
        raise ValueError     
    return sample

def revert_preprocessing(data_tensor, name):
    """
    unnormalize the data tensor by multiplying the standard deviation and adding the mean.
    args:
        data_tensor: input data tensor
        name: name of the dataset
    return:
        data_tensor: unnormalized data tensor
    """
    if name == 'mnist':
        data_tensor = data_tensor*0.3081 + 0.1307
    elif name == 'cifar10':
        data_tensor[:,0,:,:] = data_tensor[:,0,:,:]*0.2023 + 0.4914     
        data_tensor[:,1,:,:] = data_tensor[:,1,:,:]*0.1994 + 0.4822      
        data_tensor[:,2,:,:] = data_tensor[:,2,:,:]*0.2010 + 0.4465      
    else:
        raise NotImplementedError
    return data_tensor

def normalize(gradient, name):
    """
    normalize the gradient to a 0 to 1 range for display
    args:
        gradient: input gradent tensor
        name: name of the dataset
    return:
        gradient: normalized gradient tensor
    """
    if name == 'mnist':
        pass
    elif name == 'cifar10':
        # take the maximum gradient from the 3 channels
        gradient = (gradient.max(dim=1)[0]).unsqueeze(dim=1)
    # get the maximum gradient
    max_gradient = torch.max(gradient.view(len(gradient), -1), dim=1)[0]
    max_gradient = max_gradient.view(len(gradient), 1, 1, 1)
    min_gradient = torch.min(gradient.view(len(gradient), -1), dim=1)[0]
    min_gradient = min_gradient.view(len(gradient), 1, 1, 1)    
    # do normalization
    gradient = (gradient - min_gradient)/(max_gradient - min_gradient)    
    return gradient

def trace(record):
    """
    get the the path that is very likely to be visited by the input images. For each splitting node along the
    path the probability of arriving at it is also computed.
    args:
        record: record of the routing probabilities of the splitting nodes
    return:
        path: the very likely computational path
    """
    path = []
    # probability of arriving at the root node is just 1
    prob = 1
    # the starting index 
    node_idx = 1
    while node_idx < len(record):
        path.append((node_idx, prob))
        # find the children node with larger visiting probability
        if record[node_idx] >= 0.5:
            prob *= record[node_idx]
            # go to left sub-tree
            node_idx = node_idx*2
        else:
            prob *= 1 - record[node_idx]
            # go to right sub-tree
            node_idx = node_idx*2 + 1          
    return path

def get_paths(dataset, model, tree_idx, name):
    """
    compute the computational paths for the input tensors
    args:
      dataset: Pytorch dataset object
      model: pre-trained deep neural decision forest for visualizing
      tree_idx: which tree to use if there are multiple trees in the forest. 
      name: name of the dataset
   return:
      sample: randomly drawn sample
      paths: computational paths for the samples
      class_pred: model predictions for the samples
    """
    sample_num = 5
    # get some random input images
    sample = get_sample(dataset, sample_num, name)   
    # forward pass to get the routing probability
    pred, cache, _ = model(sample.cuda(), save_flag = True)
    class_pred = pred.max(dim=1)[1]
    # for now use the first tree by cache[0]
    # please refer to ndf.py if you are interested in how the forward pass is implemented
    decision = cache[0]['decision'].data.cpu().numpy()
    paths = []
    # trace the computational path for every input image
    for sample_idx in range(len(decision)):
        paths.append(trace(decision[sample_idx, :]))
    return sample, paths, class_pred

def get_node_saliency_map(dataset, model, tree_idx, node_idx, name):
  """
  get decision saliency maps for one specific splitting node
  args:
    dataset: Pytorch dataset object
    model: pre-trained neural decision forest to visualize
    tree_idx: index of the tree
    node_idx: index of the splitting node
    name: name of the dataset
  return:
    gradient: computed decision saliency maps
  """
    # pick some samples from the dataset
    sample_num = 5
    sample = get_sample(dataset, sample_num, name)
    # For now only GPU code is supported
    sample = sample.cuda()
    # enable the gradient computation (the input tensor will requires gradient computation in the backward computational graph) 
    sample.requires_grad = True
    # get the feature vectors for the drawn samples
    feats = model.feature_layer(sample)
    # using_idx gives the indices of the neurons in the last FC layer that are used to compute routing probabilities 
    using_idx = model.forest.trees[tree_idx].using_idx[node_idx + 1]
#    for sample_idx in range(len(feats)):
#        feats[sample_idx, using_idx].backward(retain_graph=True)
    # equivalent to the above commented one
    feats[:, using_idx].sum(dim = 0).backward()
    # get the gradient data
    gradient = sample.grad.data
    # get the magnitude
    gradient = torch.abs(gradient)
    # normalize the gradient for visualizing
    gradient = normalize(gradient, name)
    # plot the input data and their corresponding decison saliency maps
    plt.figure()
    # unnormalize the images for display
    sample = revert_preprocessing(sample, name)
    # plot for every input image
    for sample_idx in range(sample_num):
        plt.subplot(2, sample_num, sample_idx + 1)
        sample_to_show = sample[sample_idx].squeeze().data.cpu().numpy()
        if name == 'cifar10':
            # re-order the channels
            sample_to_show = sample_to_show.transpose((1,2,0))
            plt.imshow(sample_to_show)
        elif name == 'mnist':
            plt.imshow(sample_to_show, cmap='gray')
        else:
            raise NotImplementedError
        plt.subplot(2, sample_num, sample_idx + 1 + sample_num)
        plt.imshow(gradient[sample_idx].squeeze().cpu().numpy())
    plt.axis('off')
    plt.show()
    return gradient

def get_map(model, sample, node_idx, tree_idx, name):
"""
helper function for computing the saliency map for a specified sample and splitting node
args:
    model: pre-trained neural decison forest to visualize
    sample: input image tensors
    node_idx: index of the splitting node
    tree_idx: index of the decison tree
    name:name of the dataset
return:
    saliency_map: computed decision saliency map
"""
    # move to GPU
    sample = sample.unsqueeze(dim=0).cuda()
    # enable gradient computation for the input tensor
    sample.requires_grad = True
    # get feature vectors of the input samples
    feat = model.feature_layer(sample)
    # using_idx gives the indices of the neurons in the last FC layer that are used to compute routing probabilities 
    using_idx = model.forest.trees[tree_idx].using_idx[node_idx]
    # compute gradient by a backward pass
    feat[:, using_idx].backward()
    # get the gradient data
    gradient = sample.grad.data
    # normalize the gradient
    gradient = normalize(torch.abs(gradient), name)
    saliency_map = gradient.squeeze().cpu().numpy()
    return saliency_map

def get_path_saliency(samples, paths, class_pred, model, tree_idx, name, orientation = 'horizontal'):
"""  
show the saliency maps for the input samples with their pre-computed computational paths 
args:
  samples: input image tensor
  paths: pre-computed computational paths for the inputs
  class_pred: model predictons for the inputs
  model: pre-trained neural decison forest
  tree_idx: index of the decision tree
  name: name of the dataset
  orientation: layout of the figure
"""
    #plt.ioff()
    # plotting parameters
    plt.figure(figsize=(20,5))
    plt.rcParams.update({'font.size': 12})
    # number of input samples
    num_samples = len(samples)
    # length of the computational path
    path_length = len(paths[0])
    # iterate for every input sample
    for sample_idx in range(num_samples):
        sample = samples[sample_idx]
        # plot the sample
        plt.subplot(num_samples, path_length + 1, sample_idx*(path_length + 1) + 1)
        # unnormalize the input
        sample_to_plot = revert_preprocessing(sample.unsqueeze(dim=0), name)
        if name == 'mnist':
            plt.imshow(sample_to_plot.squeeze().cpu().numpy(), cmap='gray')
            pred_class_name = str(int(class_pred[sample_idx]))
        else:
            plt.imshow(sample_to_plot.squeeze().cpu().numpy().transpose((1,2,0)))            
            pred_class_name = cifar10_class_name[int(class_pred[sample_idx])]
        plt.axis('off')        
        plt.title('Pred:{:s}'.format(pred_class_name))
        # computational path for this sample
        path = paths[sample_idx]
        for node_idx in range(path_length):
            # compute and plot decison saliency map for each splitting node along the path
            node = path[node_idx][0]
            # probability of arriving at this node
            prob = path[node_idx][1]     
            # compute the saliency map
            saliency_map = get_map(model, sample, node, tree_idx, name)
            if orientation == 'horizontal':
                sub_plot_idx = sample_idx*(path_length + 1) + node_idx + 2
                plt.subplot(num_samples, path_length + 1, sub_plot_idx)
            elif orientation == 'vertical':
                raise NotImplementedError             
            else:
                raise NotImplementedError
            plt.imshow(saliency_map)
            plt.title('(N{:d}, P{:.2f})'.format(node, prob))
            plt.axis('off')
        # draw some arrows 
        for arrow_idx in range(num_samples*(path_length + 1) - 1):
            if (arrow_idx+1) % (path_length+1) == 0 and arrow_idx != 0:
                continue
            ax1 = plt.subplot(num_samples, path_length + 1, arrow_idx + 1)
            ax2 = plt.subplot(num_samples, path_length + 1, arrow_idx + 2)
            arrow = ConnectionPatch(xyA=[1.1,0.5], xyB=[-0.1, 0.5], coordsA='axes fraction', coordsB='axes fraction',
                      axesA=ax1, axesB=ax2, arrowstyle="fancy")
            ax1.add_artist(arrow)
    left  = 0  # the left side of the subplots of the figure
    right = 1    # the right side of the subplots of the figure
    bottom = 0.01   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.0 # the amount of width reserved for space between subplots,
                   # expressed as a fraction of the average axis width
    hspace = 0.4   # the amount of height reserved for space between subplots,
                   # expressed as a fraction of the average axis height  
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.show()
    # save figure if you need
    #plt.savefig('saved_fig.png',dpi=1200)
    return
