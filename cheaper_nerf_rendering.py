"""
rendering.py - Volume Rendering Functions for Cheaper-NeRF
Implements optimized volume rendering with zero-density filtering.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple, Any


def raw2outputs(raw: tf.Tensor,
                z_vals: tf.Tensor,
                rays_d: tf.Tensor,
                raw_noise_std: float = 0.,
                white_bkgd: bool = False,
                apply_filter: bool = True) -> Dict[str, tf.Tensor]:
    """
    Convert raw network output to rendered values.
    Implements volume rendering equation with Cheaper-NeRF optimizations.
    
    Args:
        raw: Network output [N_rays, N_samples, 4] (RGB + density)
        z_vals: Sample distances [N_rays, N_samples]
        rays_d: Ray directions [N_rays, 3]
        raw_noise_std: Noise to add to density during training
        white_bkgd: Whether to use white background
        apply_filter: Whether to apply zero-density filtering
        
    Returns:
        Dictionary containing:
            - rgb_map: Rendered RGB image [N_rays, 3]
            - depth_map: Rendered depth map [N_rays]
            - disp_map: Disparity map [N_rays]
            - acc_map: Accumulated opacity [N_rays]
            - weights: Sample weights [N_rays, N_samples]
    """
    # Extract RGB and density
    rgb = tf.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    sigma = tf.nn.relu(raw[..., 3])  # [N_rays, N_samples]
    
    # CHEAPER-NERF OPTIMIZATION: Filter zero densities
    if apply_filter:
        density_mask = sigma > 1e-10
        sigma = sigma * tf.cast(density_mask, sigma.dtype)
    
    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = tf.concat([dists, tf.broadcast_to([1e10], 
                                              shape=dists[..., :1].shape)], axis=-1)
    
    # Account for non-unit ray directions
    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)
    
    # Add noise to regularize during training
    if raw_noise_std > 0.:
        noise = tf.random.normal(sigma.shape) * raw_noise_std
        sigma = sigma + noise
    
    # Compute alpha values (opacity)
    alpha = 1. - tf.exp(-sigma * dists)
    
    # Compute transmittance (accumulated transparency)
    # T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
    transmittance = tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)
    
    # Compute weights for each sample
    # w_i = T_i * (1 - exp(-sigma_i * delta_i))
    weights = alpha * transmittance
    
    # Compute RGB map (expected color)
    rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)  # [N_rays, 3]
    
    # Compute depth map (expected distance)
    depth_map = tf.reduce_sum(weights * z_vals, axis=-1)  # [N_rays]
    
    # Compute disparity map (inverse depth)
    disp_map = 1. / tf.maximum(1e-10, 
                               depth_map / tf.reduce_sum(weights, axis=-1))
    
    # Compute accumulated opacity
    acc_map = tf.reduce_sum(weights, axis=-1)  # [N_rays]
    
    # Apply white background if needed
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    
    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'alpha': alpha,
        'transmittance': transmittance
    }


def render_rays(ray_batch: tf.Tensor,
                network_fn: Any,
                network_fine: Optional[Any],
                embed_fn: Any,
                embeddirs_fn: Optional[Any],
                N_samples: int,
                retraw: bool = False,
                perturb: float = 0.,
                N_importance: int = 0,
                white_bkgd: bool = False,
                raw_noise_std: float = 0.,
                apply_cheaper_optimizations: bool = True) -> Dict[str, tf.Tensor]:
    """
    Render rays through the scene using Cheaper-NeRF optimizations.
    
    Args:
        ray_batch: Batch of rays [N_rays, 11] (o, d, near, far, viewdirs)
        network_fn: Coarse network function
        network_fine: Fine network function (optional)
        embed_fn: Positional encoding for coordinates
        embeddirs_fn: Positional encoding for view directions
        N_samples: Number of coarse samples
        retraw: Whether to return raw predictions
        perturb: Perturbation for sample positions
        N_importance: Number of fine samples
        white_bkgd: White background
        raw_noise_std: Noise standard deviation
        apply_cheaper_optimizations: Whether to apply Cheaper-NeRF optimizations
        
    Returns:
        Rendering results dictionary
    """
    # Extract ray components
    rays_o, rays_d = ray_batch[..., :3], ray_batch[..., 3:6]  # Origins and directions
    viewdirs = ray_batch[..., 8:11] if ray_batch.shape[-1] > 8 else None
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]
    
    # Sample points along rays
    t_vals = tf.linspace(0., 1., N_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    
    # Perturb sampling positions during training
    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # CHEAPER-NERF OPTIMIZATION: Apply mean sampling reduction
    if apply_cheaper_optimizations:
        from sampling import CheaperSampler, SamplingConfig
        sampler = CheaperSampler(SamplingConfig(reduction_factor=4))
        reduced = sampler.mean_sample_reduction(
            pts, 
            tf.broadcast_to(viewdirs[..., None, :], pts.shape) if viewdirs is not None else None
        )
        pts = reduced['pts']
        if viewdirs is not None:
            viewdirs_expanded = reduced['viewdirs']
    else:
        if viewdirs is not None:
            viewdirs_expanded = tf.broadcast_to(viewdirs[..., None, :], pts.shape)
    
    # Run network
    raw = run_network(pts, viewdirs_expanded if viewdirs is not None else None,
                     network_fn, embed_fn, embeddirs_fn)
    
    # Render coarse results
    rgb_map, disp_map, acc_map, weights, depth_map = render_core(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, apply_cheaper_optimizations
    )
    
    # Hierarchical sampling for fine network
    if N_importance > 0 and network_fine is not None:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        
        # Sample additional points based on coarse weights
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance,
                              det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)
        
        # Combine and sort samples
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        
        # Compute new points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        # Apply optimization to fine samples
        if apply_cheaper_optimizations:
            reduced_fine = sampler.mean_sample_reduction(
                pts,
                tf.broadcast_to(viewdirs[..., None, :], pts.shape) if viewdirs is not None else None
            )
            pts = reduced_fine['pts']
            if viewdirs is not None:
                viewdirs_expanded = reduced_fine['viewdirs']
        
        # Run fine network
        raw = run_network(pts, viewdirs_expanded if viewdirs is not None else None,
                         network_fine, embed_fn, embeddirs_fn)
        
        # Render fine results
        rgb_map, disp_map, acc_map, weights, depth_map = render_core(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, apply_cheaper_optimizations
        )
    
    # Prepare return dictionary
    ret = {
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'depth_map': depth_map,
        'weights': weights
    }
    
    if retraw:
        ret['raw'] = raw
    
    if N_importance > 0 and network_fine is not None:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
    
    return ret


def render_core(raw: tf.Tensor,
                z_vals: tf.Tensor,
                rays_d: tf.Tensor,
                raw_noise_std: float = 0.,
                white_bkgd: bool = False,
                apply_filter: bool = True) -> Tuple[tf.Tensor, ...]:
    """
    Core rendering function with Cheaper-NeRF optimizations.
    
    Returns:
        Tuple of (rgb_map, disp_map, acc_map, weights, depth_map)
    """
    outputs = raw2outputs(raw, z_vals, rays_d, raw_noise_std, 
                         white_bkgd, apply_filter)
    
    return (outputs['rgb_map'], 
            outputs['disp_map'],
            outputs['acc_map'],
            outputs['weights'],
            outputs['depth_map'])


def run_network(pts: tf.Tensor,
                viewdirs: Optional[tf.Tensor],
                network_fn: Any,
                embed_fn: Any,
                embeddirs_fn: Optional[Any],
                netchunk: int = 1024*64) -> tf.Tensor:
    """
    Run network on batch of points with optional batching.
    
    Args:
        pts: 3D points [N_rays, N_samples, 3]
        viewdirs: View directions [N_rays, N_samples, 3]
        network_fn: Network function
        embed_fn: Positional encoding for points
        embeddirs_fn: Positional encoding for directions
        netchunk: Chunk size for batched processing
        
    Returns:
        Raw network outputs [N_rays, N_samples, 4]
    """
    # Flatten inputs
    pts_flat = tf.reshape(pts, [-1, pts.shape[-1]])
    
    # Apply positional encoding
    embedded = embed_fn(pts_flat)
    
    # Handle view directions
    if viewdirs is not None:
        viewdirs_flat = tf.reshape(viewdirs, [-1, viewdirs.shape[-1]])
        embedded_dirs = embeddirs_fn(viewdirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)
    
    # Process in chunks to save memory
    outputs_flat = batchify(lambda x: network_fn(x), netchunk)(embedded)
    
    # Reshape back
    outputs = tf.reshape(outputs_flat, 
                         list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    
    return outputs


def batchify(fn: Any, chunk: int) -> Any:
    """
    Create batched version of function for memory efficiency.
    
    Args:
        fn: Function to batchify
        chunk: Batch size
        
    Returns:
        Batched function
    """
    if chunk is None:
        return fn
    
    def ret(inputs):
        outputs = []
        for i in range(0, inputs.shape[0], chunk):
            outputs.append(fn(inputs[i:i+chunk]))
        return tf.concat(outputs, 0)
    
    return ret


def sample_pdf(bins: tf.Tensor,
               weights: tf.Tensor,
               N_samples: int,
               det: bool = False) -> tf.Tensor:
    """
    Sample from probability distribution for importance sampling.
    
    Args:
        bins: Bin edges [N_rays, N_bins]
        weights: Weights/probabilities [N_rays, N_bins]
        N_samples: Number of samples to draw
        det: Deterministic sampling
        
    Returns:
        Samples [N_rays, N_samples]
    """
    # Normalize weights
    weights = weights + 1e-5
    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    cdf = tf.cumsum(pdf, axis=-1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)
    
    # Generate uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])
    
    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds - 1)
    above = tf.minimum(cdf.shape[-1] - 1, inds)
    inds_g = tf.stack([below, above], axis=-1)
    
    # Gather values
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    
    # Linear interpolation
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


class CheaperVolumeRenderer:
    """
    Optimized volume renderer for Cheaper-NeRF with advanced features.
    """
    
    def __init__(self,
                 enable_filtering: bool = True,
                 enable_mean_sampling: bool = True,
                 filtering_threshold: float = 1e-10,
                 sampling_reduction: int = 4):
        self.enable_filtering = enable_filtering
        self.enable_mean_sampling = enable_mean_sampling
        self.filtering_threshold = filtering_threshold
        self.sampling_reduction = sampling_reduction
        
        # Statistics tracking
        self.stats = {
            'rays_rendered': 0,
            'samples_processed': 0,
            'samples_filtered': 0,
            'render_time': 0.0
        }
    
    def render(self,
               rays_o: tf.Tensor,
               rays_d: tf.Tensor,
               viewdirs: Optional[tf.Tensor],
               network_fn: Any,
               embed_fn: Any,
               embeddirs_fn: Optional[Any],
               z_vals: tf.Tensor,
               raw_noise_std: float = 0.,
               white_bkgd: bool = False) -> Dict[str, tf.Tensor]:
        """
        Render rays with Cheaper-NeRF optimizations.
        
        Returns:
            Rendering results dictionary
        """
        import time
        start_time = time.time()
        
        # Update statistics
        self.stats['rays_rendered'] += rays_o.shape[0]
        
        # Compute 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        original_samples = pts.shape[-2]
        
        # Apply mean sampling if enabled
        if self.enable_mean_sampling:
            from sampling import CheaperSampler, SamplingConfig
            sampler = CheaperSampler(
                SamplingConfig(reduction_factor=self.sampling_reduction)
            )
            reduced = sampler.mean_sample_reduction(pts, viewdirs)
            pts = reduced['pts']
            if viewdirs is not None:
                viewdirs = reduced['viewdirs']
        
        # Run network
        raw = run_network(pts, viewdirs, network_fn, embed_fn, embeddirs_fn)
        
        # Apply filtering if enabled
        if self.enable_filtering:
            sigma = tf.nn.relu(raw[..., 3])
            mask = sigma > self.filtering_threshold
            filtered_count = tf.reduce_sum(tf.cast(~mask, tf.int32))
            self.stats['samples_filtered'] += filtered_count.numpy()
            
            # Zero out filtered samples
            raw = raw * tf.cast(mask[..., None], raw.dtype)
        
        # Render
        outputs = raw2outputs(raw, z_vals, rays_d, raw_noise_std, 
                            white_bkgd, self.enable_filtering)
        
        # Update statistics
        self.stats['samples_processed'] += original_samples * rays_o.shape[0]
        self.stats['render_time'] += time.time() - start_time
        
        return outputs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        stats = self.stats.copy()
        if stats['rays_rendered'] > 0:
            stats['avg_samples_per_ray'] = stats['samples_processed'] / stats['rays_rendered']
            stats['filter_ratio'] = stats['samples_filtered'] / max(1, stats['samples_processed'])
            stats['avg_render_time'] = stats['render_time'] / stats['rays_rendered']
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'rays_rendered': 0,
            'samples_processed': 0,
            'samples_filtered': 0,
            'render_time': 0.0
        }