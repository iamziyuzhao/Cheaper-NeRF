"""
sampling.py - Cheaper Sampling Strategies for Cheaper-NeRF
Implements the core innovations: mean sampling and zero-density filtering.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    reduction_factor: int = 4          # Combine every N points
    density_threshold: float = 1e-10   # Minimum density to keep
    adaptive_threshold: bool = True    # Use adaptive thresholding
    gradient_aware: bool = False       # Preserve high-gradient regions
    importance_weight: float = 0.5     # Weight for importance sampling
    
    
class CheaperSampler:
    """
    Core sampling innovations of Cheaper-NeRF.
    Reduces computational cost through intelligent sampling.
    """
    
    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self.stats = {
            'total_samples': 0,
            'reduced_samples': 0,
            'filtered_samples': 0,
            'reduction_ratio': 0.0
        }
    
    def mean_sample_reduction(self, 
                            pts: tf.Tensor,
                            viewdirs: Optional[tf.Tensor] = None,
                            features: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        KEY INNOVATION #1: Combine multiple samples through averaging.
        
        This reduces the number of network evaluations by factor of reduction_factor.
        Mathematically: p_combined = (1/N) * Σ p_i
        
        Args:
            pts: Points tensor [..., N, 3]
            viewdirs: View directions [..., N, 3] (optional)
            features: Additional features [..., N, F] (optional)
            
        Returns:
            Dictionary with reduced tensors
        """
        original_shape = tf.shape(pts)
        N = original_shape[-2]
        reduction_factor = self.config.reduction_factor
        
        # Record statistics
        self.stats['total_samples'] += N
        
        # Calculate padding needed
        remainder = N % reduction_factor
        if remainder != 0:
            pad_count = reduction_factor - remainder
            padding = [[0, 0]] * (len(pts.shape) - 2) + [[0, pad_count], [0, 0]]
            pts = tf.pad(pts, padding, mode='EDGE')
            
            if viewdirs is not None:
                viewdirs = tf.pad(viewdirs, padding, mode='EDGE')
            if features is not None:
                features = tf.pad(features, padding, mode='EDGE')
        
        # Reshape for grouping
        new_N = tf.shape(pts)[-2] // reduction_factor
        pts_grouped = tf.reshape(pts, 
                                [..., new_N, reduction_factor, pts.shape[-1]])
        
        # Apply mean reduction
        pts_reduced = tf.reduce_mean(pts_grouped, axis=-2)
        
        # Record reduced samples
        self.stats['reduced_samples'] += new_N
        
        result = {'pts': pts_reduced}
        
        # Handle optional tensors
        if viewdirs is not None:
            viewdirs_grouped = tf.reshape(viewdirs,
                                         [..., new_N, reduction_factor, viewdirs.shape[-1]])
            result['viewdirs'] = tf.reduce_mean(viewdirs_grouped, axis=-2)
        
        if features is not None:
            features_grouped = tf.reshape(features,
                                         [..., new_N, reduction_factor, features.shape[-1]])
            result['features'] = tf.reduce_mean(features_grouped, axis=-2)
        
        # Calculate reduction ratio
        self.stats['reduction_ratio'] = 1.0 - (new_N / N)
        
        return result
    
    def filter_zero_density(self,
                           pts: tf.Tensor,
                           sigma: tf.Tensor,
                           rgb: Optional[tf.Tensor] = None,
                           features: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        KEY INNOVATION #2: Remove points with zero/near-zero density.
        
        Points with σ ≈ 0 don't contribute to the final image.
        This filtering reduces unnecessary computations.
        
        Args:
            pts: Points tensor [..., N, 3]
            sigma: Density values [..., N] or [..., N, 1]
            rgb: Color values [..., N, 3] (optional)
            features: Additional features [..., N, F] (optional)
            
        Returns:
            Dictionary with filtered tensors and mask
        """
        # Ensure sigma is 1D for the last dimension
        if len(sigma.shape) > len(pts.shape) - 1:
            sigma = tf.squeeze(sigma, axis=-1)
        
        # Determine threshold
        if self.config.adaptive_threshold:
            threshold = self._compute_adaptive_threshold(sigma)
        else:
            threshold = self.config.density_threshold
        
        # Create mask for non-zero density
        mask = sigma > threshold
        
        # Record statistics
        num_kept = tf.reduce_sum(tf.cast(mask, tf.int32))
        num_total = tf.shape(sigma)[-1]
        self.stats['filtered_samples'] += (num_total - num_kept)
        
        # Apply mask efficiently
        # Note: In practice, you might want to use tf.boolean_mask or sparse ops
        result = {
            'mask': mask,
            'pts': pts,  # Keep original for now (masking applied later)
            'sigma': sigma * tf.cast(mask, sigma.dtype),
            'kept_ratio': tf.cast(num_kept, tf.float32) / tf.cast(num_total, tf.float32)
        }
        
        if rgb is not None:
            result['rgb'] = rgb * tf.cast(mask[..., None], rgb.dtype)
        
        if features is not None:
            result['features'] = features * tf.cast(mask[..., None], features.dtype)
        
        return result
    
    def _compute_adaptive_threshold(self, sigma: tf.Tensor) -> tf.Tensor:
        """
        Compute adaptive threshold based on density distribution.
        
        Args:
            sigma: Density values
            
        Returns:
            Adaptive threshold value
        """
        # Use percentile-based thresholding
        percentile = 10.0  # Keep top 90% of density values
        threshold = tfp.stats.percentile(sigma, percentile)
        
        # Ensure minimum threshold
        min_threshold = self.config.density_threshold
        threshold = tf.maximum(threshold, min_threshold)
        
        return threshold
    
    def importance_sample(self,
                         bins: tf.Tensor,
                         weights: tf.Tensor,
                         N_samples: int,
                         det: bool = False) -> tf.Tensor:
        """
        Hierarchical importance sampling for fine network.
        Focuses samples on high-weight regions.
        
        Args:
            bins: Bin edges [..., N+1]
            weights: Weights from coarse network [..., N]
            N_samples: Number of fine samples
            det: Deterministic sampling
            
        Returns:
            New sample positions [..., N_samples]
        """
        # Add small epsilon to prevent NaN
        weights = weights + 1e-5
        
        # Normalize weights to get PDF
        pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # Compute CDF
        cdf = tf.cumsum(pdf, axis=-1)
        cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)
        
        # Generate uniform samples
        if det:
            u = tf.linspace(0., 1., N_samples)
            u = tf.broadcast_to(u, [..., N_samples])
        else:
            u = tf.random.uniform([...] + [N_samples])
        
        # Invert CDF to get sample positions
        indices = tf.searchsorted(cdf, u, side='right')
        below = tf.maximum(0, indices - 1)
        above = tf.minimum(tf.shape(cdf)[-1] - 1, indices)
        
        # Gather CDF values
        indices_g = tf.stack([below, above], axis=-1)
        cdf_g = tf.gather(cdf, indices_g, axis=-1, batch_dims=len(indices_g.shape)-2)
        bins_g = tf.gather(bins, indices_g, axis=-1, batch_dims=len(indices_g.shape)-2)
        
        # Linear interpolation
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
    
    def gradient_aware_sampling(self,
                               pts: tf.Tensor,
                               sigma: tf.Tensor,
                               keep_ratio: float = 0.5) -> tf.Tensor:
        """
        Preserve samples in high-gradient regions.
        These regions often contain surface boundaries.
        
        Args:
            pts: Sample points
            sigma: Density values
            keep_ratio: Fraction of samples to keep
            
        Returns:
            Mask indicating which samples to keep
        """
        # Compute spatial gradient of density
        with tf.GradientTape() as tape:
            tape.watch(pts)
            # Simplified gradient computation
            grad = tape.gradient(sigma, pts)
        
        if grad is None:
            # Fallback to finite differences
            grad = self._finite_difference_gradient(pts, sigma)
        
        # Compute gradient magnitude
        grad_magnitude = tf.norm(grad, axis=-1)
        
        # Keep samples with high gradient
        k = tf.cast(keep_ratio * tf.cast(tf.shape(pts)[-2], tf.float32), tf.int32)
        _, top_indices = tf.nn.top_k(grad_magnitude, k)
        
        # Create mask
        mask = tf.scatter_nd(
            tf.expand_dims(top_indices, -1),
            tf.ones(k, dtype=tf.bool),
            shape=tf.shape(grad_magnitude)
        )
        
        return mask
    
    def _finite_difference_gradient(self, pts: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
        """Compute gradient using finite differences."""
        # Simplified implementation
        eps = 1e-3
        grad = []
        
        for i in range(3):  # For each spatial dimension
            pts_plus = pts.copy()
            pts_plus[..., i] += eps
            
            pts_minus = pts.copy()
            pts_minus[..., i] -= eps
            
            # Would need to evaluate network here in practice
            # This is a placeholder
            grad_i = (sigma - sigma) / (2 * eps)
            grad.append(grad_i)
        
        return tf.stack(grad, axis=-1)
    
    def get_stats(self) -> Dict[str, float]:
        """Get sampling statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.stats = {
            'total_samples': 0,
            'reduced_samples': 0,
            'filtered_samples': 0,
            'reduction_ratio': 0.0
        }


class AdaptiveSampler(CheaperSampler):
    """
    Adaptive sampling that adjusts strategy based on scene complexity.
    """
    
    def __init__(self, config: SamplingConfig = None):
        super().__init__(config)
        self.complexity_history = []
        self.adaptive_factor = 1.0
    
    def estimate_complexity(self, sigma: tf.Tensor) -> float:
        """Estimate scene complexity from density distribution."""
        # Variance in density indicates complexity
        complexity = tf.math.reduce_std(sigma)
        
        # Update history
        self.complexity_history.append(complexity.numpy())
        if len(self.complexity_history) > 100:
            self.complexity_history.pop(0)
        
        return complexity
    
    def update_adaptive_factor(self, complexity: float):
        """Update sampling parameters based on complexity."""
        # High complexity -> less reduction
        # Low complexity -> more aggressive reduction
        
        avg_complexity = np.mean(self.complexity_history) if self.complexity_history else complexity
        
        if complexity > avg_complexity * 1.5:
            # Complex region - reduce less
            self.adaptive_factor = max(0.5, self.adaptive_factor - 0.1)
        elif complexity < avg_complexity * 0.5:
            # Simple region - reduce more
            self.adaptive_factor = min(2.0, self.adaptive_factor + 0.1)
    
    def adaptive_mean_reduction(self,
                               pts: tf.Tensor,
                               sigma: tf.Tensor,
                               viewdirs: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Adaptively adjust reduction factor based on scene complexity.
        """
        complexity = self.estimate_complexity(sigma)
        self.update_adaptive_factor(complexity)
        
        # Adjust reduction factor
        adaptive_reduction = int(self.config.reduction_factor * self.adaptive_factor)
        adaptive_reduction = max(2, min(8, adaptive_reduction))  # Clamp to reasonable range
        
        # Temporarily update config
        original_factor = self.config.reduction_factor
        self.config.reduction_factor = adaptive_reduction
        
        result = self.mean_sample_reduction(pts, viewdirs)
        
        # Restore original config
        self.config.reduction_factor = original_factor
        
        result['adaptive_factor'] = self.adaptive_factor
        result['complexity'] = complexity
        
        return result


class HierarchicalSampler:
    """
    Implements hierarchical volume sampling for Cheaper-NeRF.
    Combines coarse and fine sampling strategies.
    """
    
    def __init__(self,
                 N_samples_coarse: int = 32,
                 N_samples_fine: int = 64,
                 perturb: bool = True):
        self.N_samples_coarse = N_samples_coarse
        self.N_samples_fine = N_samples_fine
        self.perturb = perturb
        
        self.coarse_sampler = CheaperSampler(
            SamplingConfig(reduction_factor=4)
        )
        self.fine_sampler = CheaperSampler(
            SamplingConfig(reduction_factor=2)  # Less aggressive for fine
        )
    
    def sample_coarse(self,
                     rays_o: tf.Tensor,
                     rays_d: tf.Tensor,
                     near: tf.Tensor,
                     far: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initial coarse sampling along rays.
        
        Returns:
            (points, z_values)
        """
        # Generate evenly spaced samples
        t_vals = tf.linspace(0., 1., self.N_samples_coarse)
        z_vals = near * (1. - t_vals) + far * t_vals
        
        # Add perturbation for training
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            
            t_rand = tf.random.uniform(tf.shape(z_vals))
            z_vals = lower + (upper - lower) * t_rand
        
        # Compute 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        # Apply mean reduction
        reduced = self.coarse_sampler.mean_sample_reduction(pts)
        
        return reduced['pts'], z_vals
    
    def sample_fine(self,
                   rays_o: tf.Tensor,
                   rays_d: tf.Tensor,
                   z_vals_coarse: tf.Tensor,
                   weights: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Fine sampling based on coarse network weights.
        
        Returns:
            (points, z_values)
        """
        # Sample additional points based on weights
        z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        z_samples = self.fine_sampler.importance_sample(
            z_vals_mid, 
            weights[..., 1:-1],
            self.N_samples_fine,
            det=(not self.perturb)
        )
        
        # Combine and sort
        z_vals_fine = tf.sort(tf.concat([z_vals_coarse, z_samples], -1), -1)
        
        # Compute 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
        
        # Apply less aggressive reduction for fine samples
        reduced = self.fine_sampler.mean_sample_reduction(pts)
        
        return reduced['pts'], z_vals_fine
    
    def get_combined_stats(self) -> Dict[str, Dict]:
        """Get statistics from both samplers."""
        return {
            'coarse': self.coarse_sampler.get_stats(),
            'fine': self.fine_sampler.get_stats()
        }
