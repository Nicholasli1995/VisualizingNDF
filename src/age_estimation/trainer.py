"""
training utilities
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
import os

import utils
# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)

def prepare_batches(model, dataset, opt):
    # prepare some feature batches for leaf node distribution update
    with torch.no_grad():
        target_batches = []
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size = opt.batch_size, 
                                                   shuffle = True, 
                                                   num_workers = opt.num_threads)
        num_batch = int(np.ceil(opt.label_batch_size/opt.batch_size))
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == num_batch:
                break
            data = batch['image']
            targets = batch['age']  
            targets = targets.view(len(targets), -1)
            if opt.cuda:
                data, targets = data.cuda(), targets.cuda()                            
            # Get feats
            feats, _ = model.feature_layer(data)
            # release data Tensor to save memory
            del data
            for tree in model.forest.trees:
                mu = tree(feats)
                # add the minimal value to prevent some numerical issue
                mu += FLT_MIN # [batch_size, n_leaf]
                # store the routing probability for each tree
                tree.mu_cache.append(mu)
            # release memory
            del feats
            # the update rule will use both the routing probability and the 
            # target values
            target_batches.append(targets)   
    return target_batches

def train(model, optim, sche, db, opt, exp_id):
    """
    Args:
        model: the model to be trained
        optim: pytorch optimizer to be used
        db : prepared torch dataset object
        opt: command line input from the user
        exp_id: experiment id
    """
    
    best_model_dir = os.path.join(opt.save_dir, str(exp_id))
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    
    # (For FG-NET only) carry out leave-one-out validation according to the list length 
    assert len(db['train']) == len(db['eval'])
    
    # record for each training experiment
    best_MAE = []
    train_set = db['train'][exp_id]
    eval_set = db['eval'][exp_id]
    eval_loss, min_MAE, _ = evaluate(model, eval_set, opt)
    # in drop out mode, each time only leaf nodes of one tree is updated
    if opt.dropout:
        current_tree = 0
    
    # save training and validation history
    if opt.history:
        train_loss_history = []
        eval_loss_history = []
        
    for epoch in range(1, opt.epochs + 1):
        # At each epoch, train the neural decision forest and update
        # the leaf node distribution separately 
        
        # Train neural decision forest
        # set the model in the training mode
        model.train()
        # data loader
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size = opt.batch_size, 
                                                   shuffle = True, 
                                                   num_workers = opt.num_threads)    
        
        for batch_idx, batch in enumerate(train_loader):
            data = batch['image']
            target = batch['age']
            target = target.view(len(target), -1)
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data, target = data.cuda(), target.cuda()                    
            # erase all computed gradient        
            optim.zero_grad()
            #prediction, decision_loss = model(data)
            
            # forward pass to get prediction
            prediction, reg_loss = model(data)

            loss = F.mse_loss(prediction, target) + reg_loss
            
            # compute gradient in the computational graph
            loss.backward()
            
            # update parameters in the model 
            optim.step()
            
            # logging
            if batch_idx % opt.report_every == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                      epoch, batch_idx * opt.batch_size, len(train_set),
                      100. * batch_idx / len(train_loader), loss.data.item()))
            # record loss
            if opt.history:
                train_loss_history.append((epoch, batch_idx, loss.data.item()))
                    
            # Update the leaf node estimation    
            if opt.leaf_node_type == 'simple' and batch_idx % opt.update_every == 0: 
                logging.info("Epoch %d : Update leaf node prediction"%(epoch))
                target_batches = prepare_batches(model, train_set, opt)
                # Update label prediction for each tree
                logging.info("Update leaf node prediction...")
                for i in range(opt.label_iter_time):
                    # prepare features from the last feature layer
                    # some cache is also stored in the forest for leaf node
                    if opt.dropout:           
                        model.forest.trees[current_tree].update_label_distribution(target_batches)
                        current_tree = (current_tree + 1)%opt.n_tree
                    else:
                        for tree in model.forest.trees:
                            tree.update_label_distribution(target_batches)
                # release cache
                for tree in model.forest.trees:   
                    del tree.mu_cache
                    tree.mu_cache = []
                        
            
            if opt.eval and batch_idx!= 0 and batch_idx % opt.eval_every == 0:
                # evaluate model
                eval_loss, MAE, CS = evaluate(model, eval_set, opt)
                # update learning rate
                sche.step(MAE.data.item())
                # record the final MAE
                if epoch == opt.epochs:
                    last_MAE = MAE
                # record the best MAE
                if MAE < min_MAE:
                    min_MAE = MAE
                # save the best model
                    model_name = opt.model_type + train_set.name
                    best_model_path = os.path.join(best_model_dir, model_name)
                    utils.save_best_model(model.cpu(), best_model_path)
                    model.cuda()
                # update log
                utils.update_log(best_model_dir, (str(MAE.data.item()), 
                                                  str(min_MAE.data.item())), 
                                                  str(CS))
                if opt.history:
                    eval_loss_history.append((epoch, batch_idx, eval_loss, MAE))
                # reset to training mode
                model.train()
        best_MAE.append(min_MAE.data.item())
    if opt.history:
        utils.save_history(np.array(train_loss_history), np.array(eval_loss_history), opt)        
    logging.info('Training finished.')
    return model, best_MAE, last_MAE

def evaluate(model, dataset, opt, report_loss = True, predict = False):
    model.eval()
    if opt.cuda:
        model.cuda()
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = opt.batch_size, 
                                         shuffle=False, 
                                         num_workers=opt.num_threads)
    eval_loss = 0
    MAE = 0   
    # used to compute cumulative score (CS)
    threshold = opt.threshold/dataset.scale_factor
    counts_below_threshold = 0
    predicted_ages = []
    for batch_idx, batch in enumerate(loader):
        data = batch['image']
        target = batch['age']
        target = target.view(len(target), -1)
        with torch.no_grad():
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            prediction, reg_loss = model(data)  
            predicted_ages += [prediction[i].data.item() for i in range(len(prediction))]
            age = prediction.view(len(prediction), -1)
            if report_loss:
                # rescale the predicted and target residual
                MAE += torch.abs((age - target)).sum(dim = 1).sum(dim = 0)
                counts_below_threshold += (torch.abs(age-target) < threshold).sum().data.item()
                eval_loss += F.mse_loss(prediction, target.view(len(target), -1), reduction='sum').data.item()    
    if report_loss and not predict:
        eval_loss = eval_loss/len(dataset)
        MAE /= len(dataset)
        MAE *= dataset.scale_factor
        CS = counts_below_threshold/len(dataset)
        logging.info('{:s} set: Average loss: {:.4f}.'.format(dataset.split, eval_loss))
        logging.info('{:s} set: Mean absolute error: {:.4f}.'.format(dataset.split, MAE))  
        logging.info('{:s} set: Cumulative score: {:.4f}.'.format(dataset.split, CS))        
    return eval_loss, MAE, CS   