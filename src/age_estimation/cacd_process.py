# data-preprocessing for CACD 
# Reference: 
# https://github.com/shamangary/Keras-MORPH2-age-estimation/blob/master/TYY_MORPH_create_db.py
import numpy as np
import cv2
import scipy.io
import imageio as io
import argparse
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import sys
import dlib
from moviepy.editor import *

def get_dic(data_list, img_path):
    # split into different individuals
    data_dic = {}
    for idx in range(len(data_list)):
        file_name = data_list[idx][:-4]
        annotation = file_name.split('_')
        age = float(annotation[0])
        identity = ''
        for i in range(1, len(annotation) - 1):
            identity += annotation[i] + ' '
        file_path = os.path.join(img_path, data_list[idx])
        assert os.path.exists(file_path), 'Image not found!'
        if identity not in data_dic:
            temp = {'path':[file_path], 
                    'age_list':[age]}
            data_dic[identity] = temp
        else:
            data_dic[identity]['path'].append(file_path)
            data_dic[identity]['age_list'].append(age)     
    return data_dic

def get_counts(data_dic):
    SUM = 0
    for key in data_dic:
        SUM += len(data_dic[key]['path'])
    return SUM

def get_data(img_path):
    # pre-process the data for CACD
    train_list = np.load('../data/CACD_split/train.npy', allow_pickle=True)
    valid_list = np.load('../data/CACD_split/valid.npy', allow_pickle=True)
    test_list = np.load('../data/CACD_split/test.npy', allow_pickle=True)
    train_dic = get_dic(train_list, img_path)
    print('Training images: %d'%get_counts(train_dic))
    valid_dic = get_dic(valid_list, img_path)
    print('Validation images: %d'%get_counts(valid_dic))
    test_dic  = get_dic(test_list, img_path)
    print('Testing images: %d'%get_counts(test_dic))
    return train_dic, valid_dic, test_dic

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = np.matrix(points1).astype(np.float64)
    points2 = np.matrix(points2).astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def normalize(landmarks):
    center = landmarks.mean(axis = 0)
    deviation = landmarks - center
    norm_fac = np.abs(deviation).max()
    normalized_lm = deviation/norm_fac
    return normalized_lm

def get_landmarks(img_name, args):
    file_path = args.annotation + img_name[:-4] + '.landmark'
    annotation = open(file_path, 'r').read().splitlines()
    num_lm = len(annotation)  
    landmarks =  np.matrix([[float(annotation[i].split(' ')[0]), 
                       float(annotation[i].split(' ')[1])] 
                        for i in range(num_lm)])
    normalized_lm = normalize(landmarks)
    return (landmarks, normalized_lm)


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file", default='../data/CACD2000_processed')
    parser.add_argument("--img_size", type=int, default=256,
                        help="output image size")
    parser.add_argument("-annotation", type=str,
                        help="path to .landmark files", default='../data/CACD_landmark/landmark/')    
    args = parser.parse_args()
    return args

def get_mean_face(landmark_list, img_size):
    SUM = normalize(landmark_list[0][0])
    for i in range(1, len(landmark_list)):
        SUM += normalize(landmark_list[i][0])
    normalized_mean_face = SUM/len(landmark_list)
    face_size = img_size*0.3
    return normalized_mean_face*face_size + 0.5*img_size

def process_split_file():
    file_path = '/home/nicholas/Documents/Project/DRF_Age_Estimation/data/CACD_split/'
    train = open(file_path + 'train.txt', 'r').read().splitlines()
    valid = open(file_path + 'valid.txt', 'r').read().splitlines()
    test = open(file_path + 'test.txt', 'r').read().splitlines()    
    return train, valid, test

def main():
    args = get_args()
    output_path = args.output
    img_size = args.img_size

    mypath = '../data/CACD2000'
    isPlot = False
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#    landmark_list = []
#    for i in tqdm(range(len(onlyfiles))):
#        landmark_list.append(get_landmarks(onlyfiles[i], args))

    landmark_ref = np.matrix(np.load('../data/CACD_split/CACD_mean_face.npy', allow_pickle=True))
    
    # Points used to line up the images.
    ALIGN_POINTS = list(range(16))

    for i in tqdm(range(len(onlyfiles))):

        img_name = onlyfiles[i]        
        input_img = cv2.imread(mypath+'/'+img_name)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        landmark = get_landmarks(img_name, args)[0]
        M = transformation_from_points(landmark_ref[ALIGN_POINTS], 
                                       landmark[ALIGN_POINTS])
        input_img = warp_im(input_img, M, (256, 256, 3))
        io.imsave(args.output +'/'+ img_name, input_img)

if __name__ == '__main__':
    main()
