import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch

def get_sample(dataset, sample_num, name):
    # random seed
    #np.random.seed(2019)
    # get a batch of random sample images from the dataset
    indices = np.random.choice(list(range(len(dataset))), sample_num)
    if name in ['CACD', 'Morph', 'FGNET']:
        sample = [dataset[indices[i]]['image'].unsqueeze(0) for i in range(len(indices))]
        sample = torch.cat(sample, dim = 0)
        label = [dataset[indices[i]]['age'] for i in range(len(indices))]
    else:
        raise ValueError     
    return sample, label

def revert_preprocessing(data_tensor, name):
    if name == 'FGNET':
        data_tensor[:,0,:,:] = data_tensor[:,0,:,:]*0.218 + 0.425     
        data_tensor[:,1,:,:] = data_tensor[:,1,:,:]*0.191 + 0.342     
        data_tensor[:,2,:,:] = data_tensor[:,2,:,:]*0.182 + 0.314  
    elif name == 'CACD':
        data_tensor[:,0,:,:] = data_tensor[:,0,:,:]*0.3 + 0.432    
        data_tensor[:,1,:,:] = data_tensor[:,1,:,:]*0.264 + 0.359      
        data_tensor[:,2,:,:] = data_tensor[:,2,:,:]*0.252 + 0.32      
    elif name == 'Morph':
        data_tensor[:,0,:,:] = data_tensor[:,0,:,:]*0.281 + 0.564    
        data_tensor[:,1,:,:] = data_tensor[:,1,:,:]*0.255 + 0.521      
        data_tensor[:,2,:,:] = data_tensor[:,2,:,:]*0.246 + 0.508  
    else:
        raise NotImplementedError
    return data_tensor

def normalize(gradient, name):
    # take the maximum gradient from the 3 channels
    gradient = (gradient.max(dim=1)[0]).unsqueeze(dim=1)
    # normalize the gradient map to 0-1 range
    # get the maximum gradient
    max_gradient = torch.max(gradient.view(len(gradient), -1), dim=1)[0]
    max_gradient = max_gradient.view(len(gradient), 1, 1, 1)
    min_gradient = torch.min(gradient.view(len(gradient), -1), dim=1)[0]
    min_gradient = min_gradient.view(len(gradient), 1, 1, 1)    
    # Do normalization
    gradient = (gradient - min_gradient)/(max_gradient - min_gradient)    
    return gradient

def get_parents_path(leaf_idx):
    parent_list = []
    while leaf_idx > 1:
        parent = int(leaf_idx/2)
        parent_list = [parent] + parent_list
        leaf_idx = int(leaf_idx/2)
    return parent_list

def trace(record, mu, depth):
    # get the computational path that is most likely to visit 
    # from the forward pass record of one input sample
    path = []
    # probability of arriving at the root node  
    strongest_leaf_idx = np.argmax(mu)
    path.append((1,1))
    prob = 1
    parent_list = get_parents_path(strongest_leaf_idx + 2**depth)
    for i in range(1, len(parent_list)):
        current_idx = parent_list[i]
        parent_idx = parent_list[i-1]
        if current_idx == (parent_idx*2 + 1):
            prob *= 1 - record[parent_idx]
        else:
            prob *= record[parent_idx]
        path.append((current_idx, prob))       
    return path

def get_paths(dataset, model, name, depth, sample_num = 5):
    # compute the paths for the input
    sample, label = get_sample(dataset, sample_num, name)   
    # forward pass
    pred, _,  cache = model(sample.cuda(), save_flag = True)
    # pick the path that has the largest probability of being visited
    paths = []
    for sample_idx in range(len(sample)):
        max_prob = 0
        for tree_idx in range(len(cache)):
            decision = cache[tree_idx]['decision'].data.cpu().numpy()
            mu = cache[tree_idx]['mu'].data.cpu().numpy()
            tempt_path = trace(decision[sample_idx], mu[sample_idx], depth)
            if tempt_path[-1][1] > max_prob:
                max_prob = tempt_path[-1][1]
                best_path = tempt_path
        paths.append(best_path)
    return sample, label, paths, pred

def get_map(model, sample, node_idx, tree_idx, name):
    # helper function for computing the saliency map for a specified sample
    # and node
    sample = sample.unsqueeze(dim=0).cuda()
    sample.requires_grad = True
    feat = model.feature_layer(sample)[0]
    feature_mask = model.forest.trees[tree_idx].feature_mask.data.cpu().numpy()
    using_idx = np.argmax(feature_mask, axis=0)[node_idx]
    feat[:, using_idx].backward()
    gradient = sample.grad.data
    gradient = normalize(torch.abs(gradient), name)
    saliency_map = gradient.squeeze().cpu().numpy()
    return saliency_map

def get_path_saliency(samples, labels, paths, pred, model, tree_idx, name, orientation = 'horizontal'):
    # show the saliency maps for the input samples with their 
    # computational paths 
    plt.figure(figsize=(20,4))
    plt.rcParams.update({'font.size': 18})
    num_samples = len(samples)
    path_length = len(paths[0])
    for sample_idx in range(num_samples):
        sample = samples[sample_idx]
        # plot the sample
        plt.subplot(num_samples, path_length + 1, sample_idx*(path_length + 1) + 1)
        sample_to_plot = revert_preprocessing(sample.unsqueeze(dim=0), name)
        plt.imshow(sample_to_plot.squeeze().cpu().numpy().transpose((1,2,0)))            
        plt.axis('off')        
        plt.title('Pred:{:.2f}, GT:{:.0f}'.format(pred[sample_idx].data.item()*100,
                  labels[sample_idx]*100))
        path = paths[sample_idx]
        for node_idx in range(path_length):
            # compute and plot saliency for each node
            node = path[node_idx][0]
            # probability of arriving at this node
            prob = path[node_idx][1]            
            saliency_map = get_map(model, sample, node, tree_idx, name)
            if orientation == 'horizontal':
                sub_plot_idx = sample_idx*(path_length + 1) + node_idx + 2
                plt.subplot(num_samples, path_length + 1, sub_plot_idx)
            elif orientation == 'vertical':
                raise NotImplementedError             
            else:
                raise NotImplementedError
            plt.imshow(saliency_map,cmap='hot')
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
    left  = 0.02  # the left side of the subplots of the figure
    right = 1   # the right side of the subplots of the figure
    bottom = 0.01   # the bottom of the subplots of the figure
    top = 0.90     # the top of the subplots of the figure
    wspace = 0.20 # the amount of width reserved for space between subplots,
                   # expressed as a fraction of the average axis width
    hspace = 0.24   # the amount of height reserved for space between subplots,
                   # expressed as a fraction of the average axis height  
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.show()
    return