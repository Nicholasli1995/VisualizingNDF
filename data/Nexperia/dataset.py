# Nexperia Pytorch dataloader
import numpy as np
import os
import torch
import torch.utils.data
import imageio
import logging
import csv

image_extension = ".jpg"

class NexperiaDataset(torch.utils.data.Dataset):
    def __init__(self, root, paths, imgs, labels=None, split=None, mean=None,
                 std=None):   
        self.root = root
        self.paths = paths
        self.names = [path.split(os.sep)[-1][:-len(image_extension)] for path in paths]
        self.imgs = imgs
        if len(self.imgs.shape) == 3:
            self.imgs = np.expand_dims(self.imgs, axis=1)
        self.labels = labels
        self.split = split
        self.name = 'Nexperia'         
        logging.info('{:s} {:s} set contains {:d} images'.format(self.name, 
                     self.split, len(self.paths)))
        self.mean, self.std = self.get_stats(mean, std)                  
        self.normalize(self.mean, self.std)    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx]), self.labels[idx]
    
    def get_stats(self, mean=None, std=None, verbose=True):
        if mean is not None and std is not None:
            return mean, std
        # get normalization statistics
        if verbose:
            logging.info("Calculating normalizing statistics...")
        self.mean = np.mean(self.imgs)
        self.std = np.std(self.imgs)
        if verbose:
            logging.info("Calculation done for {:s} {:s} set.".format(self.name, 
                     self.split))        
        return self.mean, self.std
    
    def normalize(self, mean, std, verbose=True):
        if verbose:
            logging.info("Normalizing images...")
        self.imgs = (self.imgs - mean)/self.std
        if verbose:
            logging.info("Normalization done for {:s} {:s} set.".format(self.name, 
                     self.split))
        return
    
    def visualize(self, count=3):
        for idx in range(1, count+1):
            visualize_grid(imgs = self.imgs, labels=self.labels, title=self.split + str(idx))
        return
    
    def write_preds(self, preds):
        input_file = os.path.join(self.root, "template.csv")
        assert os.path.exists(input_file), "Please download the submission template."
        output_file = os.path.join(self.root, "submission.csv")
        save_csv(input_file, output_file, self.names, preds)
        np.save(os.path.join(self.root, 'submission.npy'), {'path':self.names, 'pred':preds})
        return
    
def save_csv(input_file, output_file, test_list, test_labels):
    """
    save a csv file for testing prediction which can be submitted to Kaggle competition
    """
    assert len(test_list) == len(test_labels)
    with open(input_file) as csv_file:
        with open(output_file, mode='w') as out_csv:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_writer = csv.writer(out_csv)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    csv_writer.writerow(row)
                    line_count += 1
                else:
                    # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                    image_name = row[0]
                    assert image_name in test_list, 'Missing prediction!'
                    index = test_list.index(image_name)
                    label = test_labels[index]
                    csv_writer.writerow([image_name, str(label)])
                    line_count += 1
            logging.info('Saved prediction. Processed {:d} lines.'.format(line_count))
    return
        
def visualize_grid(imgs, nrows=5, ncols=5, labels = None, title=""):
    """
    imgs: collection of images that supports indexing
    """
    import matplotlib.pyplot as plt
    assert nrows*ncols <= len(imgs), 'Not enough images'
    # chosen indices
    cis = np.random.choice(len(imgs), nrows*ncols, replace=False)    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            idx = row_idx*ncols + col_idx
            axes[row_idx][col_idx].imshow(imgs[cis[idx]])
            axes[row_idx][col_idx].set_axis_off()
            plt.show()
            if labels is not None:
                axes[row_idx][col_idx].set_title(str(labels[cis[idx]]))
    return

def load_data(folders):
    lgood = 0
    lbad = 1
    ltest = -1
    paths = []
    imgs = []
    labels = []
    for folder in folders:
        if 'good' in folder:
            label = lgood
        elif 'bad' in folder:
            label = lbad
        else:
            label = ltest
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if filename.endswith(image_extension):
                paths.append(filepath)
                img = imageio.imread(filepath)
                img = img.astype('float32') / 255.
                imgs.append(img) 
                labels.append(label)
    return np.array(paths), np.array(imgs), np.array(labels)
    
def get_datasets(opt, visualize=False):
    root = opt.nexperia_root
    train_ratio = opt.train_ratio
    dirs = {}
    dirs['good'] = os.path.join(root, 'train/good_0')
    dirs['bad'] = os.path.join(root, 'train/bad_1')
    dirs['test'] = os.path.join(root, 'test/all_tests')    
    train_paths, train_imgs, train_lbs = load_data([dirs['good'], dirs['bad']])
    test_paths, test_imgs, test_lbs = load_data([dirs['test']])
    # split the labeled data into training and evaluation set
    ntu = num_train_used = int(len(train_paths)*train_ratio)
    cis = chosen_indices = np.random.choice(len(train_paths), len(train_paths), replace=False)     
    used_paths, used_imgs, used_lbs = train_paths[cis[:ntu]], train_imgs[cis[:ntu]], train_lbs[cis[:ntu]]
    eval_paths, eval_imgs, eval_lbs = train_paths[cis[ntu:]], train_imgs[cis[ntu:]], train_lbs[cis[ntu:]]
    if opt.train_all:
        train_set = NexperiaDataset(root, train_paths, train_imgs, train_lbs, 'train')
    else:
        train_set = NexperiaDataset(root, used_paths, used_imgs, used_lbs, 'train')
    eval_set = NexperiaDataset(root, eval_paths, eval_imgs, eval_lbs, 'eval',
                               mean=train_set.mean, std=train_set.std)
    test_set = NexperiaDataset(root, test_paths, test_imgs, test_lbs, 'test',
                               mean=train_set.mean, std=train_set.std)
    if visualize:
        # visualize the images with annotation
        train_set.visualize()
        eval_set.visualize()
    return {'train':train_set, 'eval':eval_set, 'test':test_set}