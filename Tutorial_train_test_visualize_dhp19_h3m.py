#!/usr/bin/sudo python
# coding: utf-8


import experimenting
import event_library as el
import torch
from matplotlib import pyplot as plt
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d
from experimenting.utils.skeleton_helpers import Skeleton
import numpy as np
from experimenting.dataset.factory import Joints3DConstructor,BaseDataFactory, SimpleReadConstructor
import experimenting.utils.visualization as viz
from experimenting.utils import utilities
import time
from random import randint
import pickle

dockerMod = False

devMod = True # Switch to False to run in Docker
hw = el.utils.get_hw_property('dvs')



# Wrapper for DHP19
# dhpcore = experimenting.dataset.DHP19Core('test', base_path='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', data_dir='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', joints_dir='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])
if devMod:
    dhpcore = experimenting.dataset.DHP19Core('test', data_dir='/media/ggoyal/Shared/data/dhp19_sample/time_count_dataset/movements_per_frame', joints_dir="", hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])
    # dhpcore = experimenting.dataset.DHP19Core('test', base_path='/media/iiticubns010/Datasets/DHP19/Raw_Data/Results_and_stuff/time_count_dataset', data_dir='/media/iiticubns010/Datasets/DHP19/Raw_Data/Results_and_stuff/time_count_dataset/movements_per_frame', joints_dir='/media/iiticubns010/Datasets/DHP19/Raw_Data/Results_and_stuff/time_count_dataset/labels_full_joints/', hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])
# /media/ggoyal/Shared/data/dhp19_sample/time_count_dataset
else:
    dhpcore = experimenting.dataset.DHP19Core('test', data_dir='/media/ggoyal/Shared/data/dhp19_sample/time_count_dataset/movements_per_frame', joints_dir="/media/ggoyal/Shared/data/dhp19_sample/time_count_dataset/labels_full_joints/", hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])

if dockerMod:
    dhpcore = experimenting.dataset.DHP19Core('test', base_path='/data/DHP19/time_count_dataset', data_dir='/data/DHP19/time_count_dataset/movements_per_frame', joints_dir='/data/DHP19/time_count_dataset/labels_full_joints/', hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])

print(len(dhpcore.file_paths))

# ## Evaluate a model


if dockerMod:
    path = "/data/checkpoint"
    resultsPath = "/data/DHP19/3dpose_results"
    factory = Joints3DConstructor()
    factory.set_dataset_core(dhpcore)
    train, val, test = factory.get_datasets({'apply':{}}, {'apply':{}})
else:
    path = "/media/ggoyal/Shared/data/checkpoint_dhp19"
    resultsPath = "/media/ggoyal/Shared/data/dhp19_sample/outputs/"

model = utilities.load_model(path, "MargiposeEstimator", core=dhpcore).eval().double()


if devMod:
    factory = SimpleReadConstructor()#Joints3DConstructor()
    factory.set_dataset_core(dhpcore)
    bla= factory.get_dataset()
else:
    factory = Joints3DConstructor()
    factory.set_dataset_core(dhpcore)
    bla, val, test = factory.get_datasets({'apply':{}}, {'apply':{}})

# print(train.x_indexes)
# print(val.x_indexes)
# print(test.x_indexes)
loader = iter(torch.utils.data.DataLoader(bla, batch_size=1, shuffle=True))

b_x = next(loader)
for i in range(5):
    # some_name = randint(10000, 99999)
    start = time.time()
    b_x = next(loader)
    preds, outs = model(b_x.permute(0, -1, 1, 2))
    print(torch.squeeze(preds).size())
    print(b_x.size())
    model_runtime = time.time()
    # pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(260, 346, camera=b_y['camera'][0], z_ref=b_y['z_ref'][0]).reproject_onto_world(b_y['M'][0])
    with open(f'{resultsPath}/Example_{i}_output.pickle', 'wb') as f:
        pickle.dump(torch.squeeze(preds), f)
    with open(f'{resultsPath}/Example_{i}_input.pickle', 'wb') as f:
        pickle.dump(torch.squeeze(b_x), f)
#
# normalization_time = time.time()
# plot_skeleton_3d(Skeleton(b_y['xyz'][0]), pred_sk, fname=f"{resultsPath}/{some_name}_3d.png")
# gt_joints = torch.stack([b_y['2d_joints'][0][:, 0], b_y['2d_joints'][0][:, 1]], 1)
# pred_joints = pred_sk.get_2d_points(260, 346, extrinsic_matrix=b_y['M'][0], intrinsic_matrix=b_y['camera'][0])
#
# pred_joints = np.stack([pred_joints[:, 0], pred_joints[:, 1]], 1)
# plot_skeleton_2d(b_x[0].squeeze(), gt_joints, pred_joints,fname=f"{resultsPath}/{some_name}_2d.png")
# end = time.time()
# print(f"Runtime of the program is {end - start}")
    print(f"The execution time of the model is {model_runtime - start}")
# print(f"Time taken to predict, reproject and normalize is {normalization_time - start}")
#
