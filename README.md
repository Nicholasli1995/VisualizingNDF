# VisualizingNDF
The repository contains pre-processed images, pre-trained models and code for visualizing deep neural decisoin forest: 
1. The classification models for MNIST and CIFAR-10 are released. 
2. Pre-processed CACD dataset is released.
2. A model trained for CACD dataset is released, which achieved state-of-the-art accuracy and comsumes less memory. 

Finally, this repository is version 1.0 with more comments to write.

## Decision making for Image Classification
<div align="center">
    <img src="images/mnist_results.png">
</div>
<div align="center">
    <img src="images/cifar10_results.png">
</div>

## Decision making for Facial Age Estimation
Note how the irrelevant texture (e.g. hair) is ignored by the model during the decision making process.
<div align="center">
    <img src="images/cacd_final1.png">
</div>

## Accuracy on Cross-Age Celebrity Dataset (CACD)
| Model             | Error        | Memory Usage | FLOPs
| ----------------- | ----------- | ----------- | ----------- |
| [DRFs (CVPR 2018)](https://github.com/shenwei1231/caffe-DeepRegressionForests)    | 4.637      | 539.4MB | 16G
| [RNDF (Ours)](https://arxiv.org/abs/1908.10737)             | 4.595      | 112.4MB | 4G

## Requirements
* Python 3.6 (not tested for other versions)
* PyTorch 1.0 
* CUDA (Yet using CPU is possible if you want to modify the code)

## Pre-trained models
You can download the pre-trained models at https://drive.google.com/drive/folders/1DM6wVSknkYBqGf1UwHQgJNUp40sYDMrv?usp=sharing and place them in the "pre-trained" folder.

## Visualization for Image Classification
After downloading the pre-trained models, go to /src and
simply run 
```bash
python ndf_vis.py 
```
for CIFAR-10.

For MNIST, run 
```bash
python ndf_vis.py -dataset 'mnist'
```
## Visualization for Facial Age Estimation:
To visualize NDF for CACD dataset:
1. Download the pre-processed images at https://drive.google.com/file/d/1OBu62cpnaMl5EX8EsjfEenRVv9rk3trt/view?usp=sharing and decompress it into the "/src/data" folder.
2. Download metadata folder at https://drive.google.com/drive/folders/1s_Ml82O4FVkC34PCE4ttrYhta3EKeYdo?usp=sharing and place it under the "/src/data" folder.
3. Go to /src/age_estimation and run
```bash
python ndf_vis.py 
```
Sorry about the hard-coded paths and few comments. Please refer to the similar classification code for detailed comments for now. Future updates will introduce more comments.

## Training for Image Classification
To train a deep neural decision forest for CIFAR-10, use 
```bash
python main.py
```
For MNIST, run 
```bash
python main.py -dataset 'mnist' -epochs 50
```

## Training for Facial Age Estimation (Beta)
To train RNDF for CACD dataset:
Follow the same step 1 and 2 as in visualization for data preparation. Finally, go to /src/age_estimation and run
```bash
python main.py -train True
```
To test a pre-trained model for CACD, go to /src/age_estimation and run
```bash
python main.py -evaluate True -test_model_path "Your Path"
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

- [Visualizing the decision-making process in deep neural decision forest](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Explainable%20AI/Li_Visualizing_the_Decision-making_Process_in_Deep_Neural_Decision_Forest_CVPRW_2019_paper.pdf)
- [Facial age estimation by deep residual decision making](https://arxiv.org/abs/1908.10737)
- [Deep neural decision forests](http://openaccess.thecvf.com/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)
