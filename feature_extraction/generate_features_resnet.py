###
# This file will:
# 1. Generate and save ResNet features in a given folder
# 2. preprocess Alexnet features using PCA and save them in another folder
###
# Run from root directory as:
#     python -m feature_extraction.generate_features_resnet [ARGS]
#
import glob
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import random
from tqdm import tqdm
import torchvision
from torchvision import transforms as trn
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA
# our imports
from data.utils import subsample
from models.resnet3d50 import ResNet3D50Backbone
from models.decoders import ClassDecoder

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

def load_resnet(model_checkpoint):
    """This function initializes a ResNet503D and  and load
    its weights from a pretrained model
    ----------
    model_checkpoint : str
        path of model checkpoint.

    Returns
    -------
    model
        pytorch model of ResNet3D50Backbone

    """


    model = ResNet3D50Backbone()
    model_file = model_checkpoint
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def sample_video_from_mp4(filename, num_frames=16):
    """This function takes a mp4 video file as input and returns
    a list of uniformly sampled frames (PIL Image).
    Parameters
    ----------
    filename : str
        path to mp4 video file
    num_frames : int
        how many frames to select using uniform frame sampling.
    Returns
    -------
    vid: the video as a PyTorch tensor
    """
    vid, _, _ = torchvision.io.read_video(filename, pts_unit='sec')
    vid = subsample(vid, num_frames)
    vid = vid.movedim(3, 0)
    return vid

def get_activations_and_save(model, video_list, activations_dir, sampling_rate = 1):
    """This function generates Alexnet features and save them in a specified directory.
    Parameters
    ----------
    model :
        pytorch model : alexnet.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    sampling_rate : int
        how many frames to skip when feeding into the network.
    """

    resize_normalize = trn.Compose([
            trn.ConvertImageDtype(torch.float32),
            trn.Resize((224,224)),
            trn.Lambda(lambda x: x.permute(1, 0, 2, 3)), # CTHW to TCHW
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            trn.Lambda(lambda x: x.permute(1, 0, 2, 3))]) # TCHW to CTHW

    for video_file in tqdm(video_list):
        vid = sample_video_from_mp4(video_file)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        vid = resize_normalize(vid).unsqueeze(0)
        activations = []
        if torch.cuda.is_available()
            vid = vid.cuda()
        x = model.features(vid)
        for i, feat in enumerate(x):
            activations.append(feat.data.cpu().numpy())
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            avg_layer_activation = activations[layer].mean((0, 2)).ravel()
            np.save(save_path,avg_layer_activation)



def do_PCA_and_save(activations_dir, save_dir):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
.

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """

    layers = ['layer_1','layer_2','layer_3','layer_4','layer_5']
    n_components = 100
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list),feature_dim.shape[0]))
        for i,activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i,:] = temp
        x_train = x[:1000,:]
        x_test = x[1000:,:]

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components,random_state=seed)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = os.path.join(save_dir,"train_"+layer)
        test_save_path = os.path.join(save_dir,"test_"+layer)
        np.save(train_save_path,x_train)
        np.save(test_save_path,x_test)

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction from 3D ResNet50 and preprocessing using PCA')
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',default = './AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir','--save_dir', help='saves processed features',default = './resnet_features', type=str)
    parser.add_argument('--ckpt', type=str, help='Path to model checkpoint to load.')
    args = vars(parser.parse_args())

    save_dir=args['save_dir']
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = args['video_data_dir']
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load 3D ResNet50
    checkpoint_path = args['ckpt']
    model = load_resnet(checkpoint_path)

    # get and save activations
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir)

    # preprocessing using PCA and save
    pca_dir = os.path.join(save_dir, 'pca_100')
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir)


if __name__ == "__main__":
    main()
