"""
utils.py - Utility functions for Cheaper-NeRF
"""

import os
import random
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
import imageio
import json


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def get_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / 1024**3
    except:
        return 0.0


def to8b(x):
    """Convert to 8-bit image."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def save_image(path: str, img: np.ndarray):
    """Save image to disk."""
    imageio.imwrite(path, to8b(img))


def load_image(path: str) -> np.ndarray:
    """Load image from disk."""
    img = imageio.imread(path)
    return img.astype(np.float32) / 255.


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between predicted and target images."""
    mse = np.mean((pred - target) ** 2)
    psnr = -10. * np.log10(mse)
    return psnr


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM between predicted and target images."""
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, target, channel_axis=2)


"""
data_loader.py - Data loading utilities for Cheaper-NeRF
"""


def load_nerf_data(datadir: str, 
                   dataset_type: str,
                   white_bkgd: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Load NeRF dataset.
    
    Args:
        datadir: Path to dataset
        dataset_type: Type of dataset ('blender', 'llff', 'deepvoxels')
        white_bkgd: Whether to use white background
        
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    if dataset_type == 'blender':
        return load_blender_data(datadir, white_bkgd)
    elif dataset_type == 'llff':
        return load_llff_data(datadir)
    elif dataset_type == 'deepvoxels':
        return load_deepvoxels_data(datadir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_blender_data(basedir: str, 
                      white_bkgd: bool = False,
                      half_res: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Load synthetic Blender dataset.
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    splits = ['train', 'val', 'test']
    all_data = {}
    
    for s in splits:
        meta_path = os.path.join(basedir, f'transforms_{s}.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        imgs = []
        poses = []
        
        # Skip some frames for faster loading in reduced mode
        skip = 1
        if s == 'train' and half_res:
            skip = 2
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)
            
            # Resize if needed
            if half_res:
                import cv2
                H, W = img.shape[:2]
                img = cv2.resize(img, (W//2, H//2), cv2.INTER_AREA)
            
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = np.array(imgs).astype(np.float32) / 255.
        poses = np.array(poses).astype(np.float32)
        
        # Handle alpha channel
        if imgs.shape[-1] == 4:
            if white_bkgd:
                # Composite onto white background
                imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
            else:
                imgs = imgs[..., :3]
        
        # Camera parameters
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        data = {
            'images': imgs,
            'poses': poses,
            'hwf': [H, W, focal],
            'near': 2.,
            'far': 6.
        }
        
        all_data[s] = data
    
    return all_data['train'], all_data['val'], all_data['test']


def load_llff_data(basedir: str,
                   factor: int = 8,
                   recenter: bool = True,
                   bd_factor: float = .75) -> Tuple[Dict, Dict, Dict]:
    """
    Load LLFF real-world dataset.
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Simplified placeholder - would need full LLFF loading logic
    poses_bounds = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    
    # Load images
    imgdir = os.path.join(basedir, 'images')
    if factor > 1:
        imgdir = os.path.join(basedir, f'images_{factor}')
    
    imgfiles = sorted(os.listdir(imgdir))
    imgs = []
    for f in imgfiles:
        if f.endswith('.JPG') or f.endswith('.jpg') or f.endswith('.png'):
            img = imageio.imread(os.path.join(imgdir, f))
            imgs.append(img)
    
    imgs = np.array(imgs).astype(np.float32) / 255.
    
    # Split data
    n_imgs = len(imgs)
    i_test = np.arange(n_imgs)[::8]
    i_val = i_test
    i_train = np.array([i for i in range(n_imgs) if i not in i_test])
    
    # Prepare data dictionaries
    H, W = imgs[0].shape[:2]
    focal = poses[0, -1, -1]
    
    train_data = {
        'images': imgs[i_train],
        'poses': poses[i_train, :3, :4],
        'hwf': [H, W, focal],
        'near': np.min(bounds) * bd_factor,
        'far': np.max(bounds)
    }
    
    val_data = {
        'images': imgs[i_val],
        'poses': poses[i_val, :3, :4],
        'hwf': [H, W, focal],
        'near': np.min(bounds) * bd_factor,
        'far': np.max(bounds)
    }
    
    test_data = {
        'images': imgs[i_test],
        'poses': poses[i_test, :3, :4],
        'hwf': [H, W, focal],
        'near': np.min(bounds) * bd_factor,
        'far': np.max(bounds)
    }
    
    return train_data, val_data, test_data


def load_deepvoxels_data(basedir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load DeepVoxels dataset.
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Placeholder implementation
    print(f"Loading DeepVoxels data from {basedir}")
    
    # Would need actual DeepVoxels loading implementation
    # For now, return dummy data
    dummy_data = {
        'images': np.random.rand(10, 512, 512, 3).astype(np.float32),
        'poses': np.random.rand(10, 3, 4).astype(np.float32),
        'hwf': [512, 512, 500.0],
        'near': 0.5,
        'far': 3.5
    }
    
    return dummy_data, dummy_data, dummy_data


def generate_rays(H: int, W: int, focal: float, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate camera rays for a given camera pose.
    
    Args:
        H: Image height
        W: Image width
        focal: Focal length
        c2w: Camera-to-world transformation matrix
        
    Returns:
        Tuple of (rays_origin, rays_direction)
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    
    dirs = np.stack([(i - W * .5) / focal,
                     -(j - H * .5) / focal,
                     -np.ones_like(i)], -1)
    
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    
    return rays_o, rays_d


def get_rays_batch(H: int, W: int, focal: float, c2w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate rays for batched camera poses.
    
    Args:
        H: Image height  
        W: Image width
        focal: Focal length
        c2w: Batch of camera poses [B, 3, 4]
        
    Returns:
        Tuple of (rays_origin [B, H, W, 3], rays_direction [B, H, W, 3])
    """
    batch_size = tf.shape(c2w)[0]
    
    # Generate pixel coordinates
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    
    # Camera space directions
    dirs = tf.stack([(i - W * .5) / focal,
                     -(j - H * .5) / focal,
                     -tf.ones_like(i)], -1)
    
    # Expand for batch dimension
    dirs = tf.expand_dims(dirs, 0)  # [1, H, W, 3]
    dirs = tf.tile(dirs, [batch_size, 1, 1, 1])  # [B, H, W, 3]
    
    # Transform to world space
    # [B, H, W, 3, 1] x [B, 1, 1, 3, 3] -> [B, H, W, 3]
    rays_d = tf.reduce_sum(dirs[..., None, :] * c2w[:, None, None, :3, :3], axis=-1)
    
    # Broadcast ray origins
    rays_o = tf.broadcast_to(c2w[:, None, None, :3, -1], tf.shape(rays_d))
    
    return rays_o, rays_d


class RayBatcher:
    """
    Utility class for batching rays during training.
    """
    
    def __init__(self, rays: np.ndarray, rgb: np.ndarray, batch_size: int):
        self.rays = rays
        self.rgb = rgb
        self.batch_size = batch_size
        self.n_rays = rays.shape[0]
        self.curr_idx = 0
        
        # Shuffle initially
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the rays."""
        perm = np.random.permutation(self.n_rays)
        self.rays = self.rays[perm]
        self.rgb = self.rgb[perm]
        self.curr_idx = 0
    
    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch of rays."""
        start_idx = self.curr_idx
        end_idx = min(start_idx + self.batch_size, self.n_rays)
        
        batch_rays = self.rays[start_idx:end_idx]
        batch_rgb = self.rgb[start_idx:end_idx]
        
        self.curr_idx = end_idx
        
        # Reset and shuffle if we've gone through all rays
        if self.curr_idx >= self.n_rays:
            self.shuffle()
        
        return batch_rays, batch_rgb
