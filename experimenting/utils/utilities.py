import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import experimenting


def load_model(load_path: str, module: str, **kwargs):
    """
    Main function to load a checkpoint.
    Args:
        load_path: path to the checkpoint directory
        module: python module (e.g., experimenting.agents.Base)
        kwargs: arguments to override while loading checkpoint

    Returns
        Lightning module loaded from checkpoint, if exists
    """
    print("Loading training")
    load_path = get_checkpoint_path(load_path)
    print("Loading from ... ", load_path)

    if os.path.exists(load_path):

        model = getattr(experimenting.agents, module).load_from_checkpoint(
            load_path, **kwargs
        )
    else:
        raise FileNotFoundError()

    return model


def get_checkpoint_path(checkpoint_dir: str) -> str:
    # CHECKPOINT file are
    if os.path.isdir(checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
        load_path = os.path.join(checkpoint_dir, checkpoints[0])
    else:
        raise Exception("Not checkpoint dir")
    return load_path


def instantiate_new_model(
    cfg: DictConfig, core: experimenting.dataset.BaseCore
) -> pl.LightningModule:
    """
    Instantiate new module from scratch using provided `hydra` configuration
    """
    model = getattr(experimenting.agents, cfg.training.module)(
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler,
        model_zoo=cfg.model_zoo,
        core=core,
        **cfg.training
    )
    return model

def save_2D_prediction(skeleton,fname,overwrite = False):
    if os.path.splitext(fname)[1] != '.npy':
        raise Exception("Please define a numpy file.")
    if torch.is_tensor(skeleton):  # Convert the tensor to numpy array
        skeleton = torch.squeeze(skeleton).detach().numpy()

    skeleton = np.expand_dims(skeleton, 0) # Add another dim to the left for the row
    # print(skeleton.shape)

    if os.path.exists(fname) and not(overwrite):
        # print(os.path.exists(fname))
        # print(fname+" file found")
        old_array = np.load(fname, allow_pickle=True)
        # if old_array.shape[1:2]!=[13,2]:
        #     raise Exception(f"Dimension miss-match. File has skeletons shaped {old_array.shape}")
        new_array = np.concatenate((old_array,skeleton),axis= 0)
    else:
        new_array = skeleton
    np.save(fname, new_array)


def save_timestamp(t,fname,overwrite = False):
    if os.path.splitext(fname)[1] != '.npy':
        raise Exception("Please define a numpy file.")

    if os.path.exists(fname) and not(overwrite):
        # print(fname+" file found")
        old_array = np.load(fname, allow_pickle=True)
        # new_array = np.array([old_array,t])
        new_array = np.concatenate((old_array,np.array([t])),axis= 0)
    else:
        new_array = np.array([t])
    np.save(fname, new_array)
