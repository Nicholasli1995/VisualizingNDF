# VisualizingNDF
Background: Neural decision forest (NDF) [3] combines the representation power of deep neural networks (DNNs) and the divide-and-conquer idea of traditional decision trees. It conducts inference by making decisions based on image features extracted by DNNs. This decision-making process can be traced and visualized with Decision Saliency Maps (DSMs) [1], which highlight important regions of the input that influence the decision process more. 

Contents: This repository contains official Pytorch code for training and visualizing NDF. Pre-processed data and pre-trained models are also released. Both classification and regression problems are considered and specific tasks include:  
1. Image classification for MNIST, CIFAR-10 and Nexperia semiconductor. The last is a Kaggle competition organized by [MATH 6380O (Advanced Topics in Deep Learning)](https://deeplearning-math.github.io/) in HKUST. 
2. Facial age estimation on the large-scale Cross-Age Celebrity Dataset (CACD). Pre-processed data and a new model (RNDF) [2] is released. RNDF achieves state-of-the-art accuracy while comsumes less memory. 

## Example: decision-making for image classification
The left-most column shows the input images. Each row visualizes one path from the root node towards the leaf node in a soft decision tree. Each image in the row represents the DSM [1] for one splitting node, where (Na, Pb) means the input arrives at node a with probability b. For example, the input arrives at the root node with probability 1 is indicated by (N1, P1.0). Each DSM highlights the spatial region that has larger influence on the corresponding splitting node. For example, the foreground object is more important for NDF when making decisions.  
<div align="center">
    <img src="teasers/mnist_results.png">
</div>
<div align="center">
    <img src="teasers/cifar10_results.png">
</div>

## Example: decision-making for facial age estimation
Note how the irrelevant texture (e.g. hair) is ignored by NDF during its decision making process.
<div align="center">
    <img src="teasers/cacd_final1.png">
</div>

## Performance on Cross-Age Celebrity Dataset (CACD)
| Model             | Error        | Memory Usage | FLOPs
| ----------------- | ----------- | ----------- | ----------- |
| [DRFs (CVPR 2018)](https://github.com/shenwei1231/caffe-DeepRegressionForests)    | 4.637      | 539.4MB | 16G
| [RNDF (Ours)](https://arxiv.org/abs/1908.10737)             | 4.595      | 112.4MB | 4G

## Dependency
* Python 3.6 (not tested for other versions)
* PyTorch >= 1.0 
* Matplotlib
* Numpy
* CUDA (CPU mode is not implemented)

## Pre-trained models
You can download the pre-trained models [here](https://drive.google.com/drive/folders/1DM6wVSknkYBqGf1UwHQgJNUp40sYDMrv?usp=sharing) and place them in the "pre-trained" folder.

## Usage: visualizing pre-trained NDF for image classification
After downloading the pre-trained models, go to /classification and
run 
```bash
python ndf_vis.py 
```
for CIFAR-10.

For MNIST, run 
```bash
python ndf_vis.py -dataset 'mnist'
```
## Usage: visualizing pre-trained RNDF for facial age estimation
To visualize NDF for CACD dataset:
1. Download the pre-processed images [here](https://drive.google.com/file/d/1OBu62cpnaMl5EX8EsjfEenRVv9rk3trt/view?usp=sharing) and decompress it into the "/data" folder.
2. Download the metadata folder [here](https://drive.google.com/drive/folders/1s_Ml82O4FVkC34PCE4ttrYhta3EKeYdo?usp=sharing) and place it under "/data".
3. Go to /regression and run
```bash
python ndf_vis.py 
```
Please refer to the classification counterpart for detailed comments. Future updates will introduce more comments for regression.

## Usage: training NDF for image classification
To train a deep neural decision forest for CIFAR-10, go to /classification and run 
```bash
python main.py
```
For MNIST, run 
```bash
python main.py -dataset 'mnist' -epochs 50
```

## Usage: training NDF for facial age estimation
To train a RNDF (R stands for residual) for CACD dataset, follow the same step 1 and 2 as in visualization. Finally, go to /regression and run
```bash
python main.py -train True
```
To test the pre-trained model on CACD, go to /regression and run
```bash
python main.py -evaluate True -test_model_path "YourPATH/CACD_MAE_4.59.pth"
```
The released model should give a MAE of 4.59

## License
MIT

## Citation
Please consider citing the related papers in your publications if they help your research:

    @InProceedings{Li_2019_CVPR_Workshops,
    author = {Li, Shichao and Cheng, Kwang-Ting},
    title = {Visualizing the Decision-making Process in Deep Neural Decision Forest},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
    }
    
    @article{li2019facial,
      title={Facial age estimation by deep residual decision making},
      author={Li, Shichao and Cheng, Kwang-Ting},
      journal={arXiv preprint arXiv:1908.10737},
      year={2019}
    }
    
    @inproceedings{kontschieder2015deep,
      title={Deep neural decision forests},
      author={Kontschieder, Peter and Fiterau, Madalina and Criminisi, Antonio and Rota Bulo, Samuel},
      booktitle={Proceedings of the IEEE international conference on computer vision},
      pages={1467--1475},
      year={2015}
    }

Links to the papers:

1. [Visualizing the decision-making process in deep neural decision forest](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Explainable%20AI/Li_Visualizing_the_Decision-making_Process_in_Deep_Neural_Decision_Forest_CVPRW_2019_paper.pdf)
2. [Facial age estimation by deep residual decision making](https://arxiv.org/abs/1908.10737)
3. [Deep neural decision forests](http://openaccess.thecvf.com/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)
