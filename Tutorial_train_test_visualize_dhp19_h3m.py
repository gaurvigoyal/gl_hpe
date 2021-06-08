#!/usr/bin/env python
# coding: utf-8

# In[1]:


import experimenting
import event_library as el
import torch
from matplotlib import pyplot as plt
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d
from experimenting.utils.skeleton_helpers import Skeleton
import numpy as np
from experimenting.dataset.factory import Joints3DConstructor
import experimenting.utils.visualization as viz
import experimenting
from experimenting.utils import utilities
import time
from random import randint

# In[2]:

hw = el.utils.get_hw_property('dvs')

# In[ ]:


# Wrapper for Events-H3m
# h3mcore = experimenting.dataset.HumanCore('test', '/data/gscarpellini/dataset/human3.6m/constant_count', '/data/gscarpellini/dataset/human3.6m/constant_count/gt.npz', 'cross-view', 1, test_cams=[1, 3], test_subjects=[6, 7])
#

# In[3]:


# Wrapper for DHP19
# dhpcore = experimenting.dataset.DHP19Core('test', base_path='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', data_dir='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', joints_dir='T:\Datasets\DHP19\out\h5_dataset_7500_events\346x260', hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])
dhpcore = experimenting.dataset.DHP19Core('test', base_path='/mnt/iiticubns010/human_pose_estimation/Datasets/DHP19/time_count_dataset', data_dir='/mnt/iiticubns010/human_pose_estimation/Datasets/DHP19/time_count_dataset/movements_per_frame', joints_dir='/mnt/iiticubns010/human_pose_estimation/Datasets/DHP19/time_count_dataset/labels_full_joints/', hm_dir="",  labels_dir="", preload_dir="", n_joints=13, n_classes=33, partition='cross-subject', n_channels=1, cams=[1, 3], movements=None, test_subjects=[6, 7])


# ## Example using H3m

# In[5]:


# idx = 10
# print(dhpcore.frames_info[idx+1])
# sk, intr, extr = dhpcore.get_joint_from_id(idx)
# frame = dhpcore.get_frame_from_id(idx)
# joints = sk.get_2d_points(260, 346, intrinsic_matrix=intr, extrinsic_matrix=extr)
# plot_skeleton_3d(sk)
# plot_skeleton_2d(frame.squeeze(), joints)


# ## Evaluate a model 
# 

# ### H3m
# 

# In[8]:


# path = "D:\code\event-based-monocular-hpe\03-21-19-55_exp_resnet50_pretrained_True_stages_1\checkpoints"
# model = utilities.load_model(path, "MargiposeEstimator", core=h3mcore).double()


# In[9]:


# factory = Joints3DConstructor()
# factory.set_dataset_core(h3mcore)
# train, val, test = factory.get_datasets({'apply':{}}, {'apply':{}})


# In[8]:


# loader = iter(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))


# In[71]:


# viz.plot_skeleton_2d(b_x[0].squeeze(),b_y['2d_joints'][0], pred_sk.get_2d_points(260, 346, extrinsic_matrix=b_y['M'][0], intrinsic_matrix=b_y['camera'][0]))


# ### DHP19
# 

# In[9]:


path = "/mnt/Shared/code/event-based-monocular-hpe/03-21-19-55_exp_resnet50_pretrained_True_stages_1/checkpoints"
model = utilities.load_model(path, "MargiposeEstimator", core=dhpcore).eval().double()


# In[10]:


factory = Joints3DConstructor()
factory.set_dataset_core(dhpcore)
train, val, test = factory.get_datasets({'apply':{}}, {'apply':{}})


# In[11]:


loader = iter(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))


# In[12]:


#b_x, b_y = next(loader)


# In[13]:
some_name = randint(10000, 99999)
start = time.time()
b_x, b_y = next(loader)
preds, outs = model(b_x.permute(0, -1, 1, 2))
model_runtime = time.time()
pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(260, 346, camera=b_y['camera'][0], z_ref=b_y['z_ref'][0]).reproject_onto_world(b_y['M'][0])
normalization_time = time.time()
plot_skeleton_3d(Skeleton(b_y['xyz'][0]), pred_sk, fname=f"/mnt/Shared/data/gl-hpe/3dpose_results/{some_name}_3d.png")
gt_joints = torch.stack([b_y['2d_joints'][0][:, 0], b_y['2d_joints'][0][:, 1]], 1)
pred_joints = pred_sk.get_2d_points(260, 346, extrinsic_matrix=b_y['M'][0], intrinsic_matrix=b_y['camera'][0])

pred_joints = np.stack([pred_joints[:, 0], pred_joints[:, 1]], 1)
plot_skeleton_2d(b_x[0].squeeze(), gt_joints, pred_joints,fname=f"/mnt/Shared/data/gl-hpe/3dpose_results/{some_name}_2d.png")
end = time.time()
print(f"Runtime of the program is {end - start}")
print(f"The execution time of the model is {model_runtime - start}")
print(f"Time taken to predict, reproject and normalize is {normalization_time - start}")

