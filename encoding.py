"""
encoding.py - Positional Encoding Module for Cheaper-NeRF
Implements efficient positional encoding with caching optimizations.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class EncodingConfig:
    """Configuration for positional encoding."""
    include_input: bool = True
    input_dims: int = 3
    max_freq_log2: int = 9
    num_freqs: int = 10
    log_sampling: bool = True
    periodic_fns: List[Callable] = None
    
    def __post_init__(self):
        if self.periodic_fns is None:
            self.periodic_fns = [tf.math.sin, tf.math.cos]


class PositionalEncoder:
    """
    Base class for positional encoding.
    Maps low-dimensional inputs to higher-dimensional representations.
    """
    
    def __init__(self, config: EncodingConfig):
        self.config = config
        self.embed_fns = []
        self.out_dim = 0
        self._build_embedding_functions()
    
    def _build_embedding_functions(self):
        """Construct the embedding functions."""
        d = self.config.input_dims
        out_dim = 0
        
        if self.config.include_input:
            self.embed_fns.append(lambda x: x)
            out_dim += d
        
        # Generate frequency bands
        if self.config.log_sampling:
            freq_bands = 2.**tf.linspace(0., self.config.max_freq_log2, 
                                         self.config.num_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**self.config.max_freq_log2, 
                                    self.config.num_freqs)
        
        # Create embedding functions for each frequency and periodic function
        for freq in freq_bands:
            for p_fn in self.config.periodic_fns:
                self.embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                out_dim += d
        
        self.out_dim = out_dim
    
    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply positional encoding to inputs.
        
        Args:
            inputs: Tensor of shape [..., input_dims]
        
        Returns:
            Encoded tensor of shape [..., out_dim]
        """
        return tf.concat([fn(inputs) for fn in self.embed_fns], axis=-1)


class CheaperPositionalEncoder(PositionalEncoder):
    """
    Optimized positional encoder for Cheaper-NeRF with caching.
    Reduces redundant computations through intelligent caching.
    """
    
    def __init__(self, config: EncodingConfig, cache_size: int = 10000):
        super().__init__(config)
        self.cache_size = cache_size
        self.encoding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-compute frequency multipliers for efficiency
        self._precompute_multipliers()
    
    def _precompute_multipliers(self):
        """Pre-compute frequency multipliers to avoid repeated calculations."""
        if self.config.log_sampling:
            freq_bands = 2.**tf.linspace(0., self.config.max_freq_log2, 
                                         self.config.num_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**self.config.max_freq_log2, 
                                    self.config.num_freqs)
        
        self.freq_multipliers = freq_bands
    
    def encode(self, inputs: tf.Tensor, use_cache: bool = True) -> tf.Tensor:
        """
        Apply positional encoding with optional caching.
        
        Args:
            inputs: Tensor of shape [..., input_dims]
            use_cache: Whether to use caching mechanism
        
        Returns:
            Encoded tensor of shape [..., out_dim]
        """
        if not use_cache:
            return super().encode(inputs)
        
        # Check cache for similar inputs (simplified for demonstration)
        input_key = self._get_cache_key(inputs)
        
        if input_key in self.encoding_cache:
            self.cache_hits += 1
            return self.encoding_cache[input_key]
        
        # Compute encoding
        self.cache_misses += 1
        encoded = super().encode(inputs)
        
        # Update cache with LRU-like behavior
        if len(self.encoding_cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.encoding_cache))
            del self.encoding_cache[oldest_key]
        
        self.encoding_cache[input_key] = encoded
        return encoded
    
    def _get_cache_key(self, inputs: tf.Tensor) -> str:
        """Generate a cache key for the input tensor."""
        # Simplified: hash based on tensor shape and mean
        # In practice, you'd want a more sophisticated hashing mechanism
        shape_str = str(inputs.shape.as_list())
        mean_val = tf.reduce_mean(inputs).numpy()
        return f"{shape_str}_{mean_val:.6f}"
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.encoding_cache)
        }


class IntegratedPositionalEncoder(CheaperPositionalEncoder):
    """
    Integrated positional encoding for handling uncertainties.
    Used when dealing with cone frustums instead of rays.
    """
    
    def encode_with_covariance(self, inputs: tf.Tensor, 
                               covs: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply integrated positional encoding.
        
        Args:
            inputs: Mean positions [..., input_dims]
            covs: Covariance matrices [..., input_dims, input_dims]
        
        Returns:
            Integrated encoded features
        """
        if covs is None:
            return self.encode(inputs)
        
        # Compute expected values under Gaussian distribution
        encoded_list = []
        
        if self.config.include_input:
            encoded_list.append(inputs)
        
        # For each frequency band
        for freq in self.freq_multipliers:
            # Compute variance of the encoding
            var = tf.linalg.diag_part(covs) * (freq ** 2)
            
            # Apply integrated encoding formula
            # E[sin(freq * x)] under N(mu, sigma^2) 
            # = sin(freq * mu) * exp(-0.5 * freq^2 * sigma^2)
            damping = tf.exp(-0.5 * var)
            
            for p_fn in self.config.periodic_fns:
                encoded = p_fn(inputs * freq) * damping
                encoded_list.append(encoded)
        
        return tf.concat(encoded_list, axis=-1)


class HashEncoder:
    """
    Multi-resolution hash encoding for faster convergence.
    Alternative to positional encoding for Cheaper-NeRF.
    """
    
    def __init__(self, 
                 num_levels: int = 16,
                 features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 finest_resolution: int = 512):
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Calculate growth factor
        self.b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) 
                       / (num_levels - 1))
        
        self.out_dim = num_levels * features_per_level
        
        # Initialize hash tables
        self._initialize_hash_tables()
    
    def _initialize_hash_tables(self):
        """Initialize multi-resolution hash tables."""
        self.hash_tables = []
        table_size = 2 ** self.log2_hashmap_size
        
        for level in range(self.num_levels):
            # Each level has its own hash table
            hash_table = tf.Variable(
                tf.random.normal([table_size, self.features_per_level], 
                               stddev=1e-4),
                trainable=True,
                name=f'hash_table_level_{level}'
            )
            self.hash_tables.append(hash_table)
    
    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply multi-resolution hash encoding.
        
        Args:
            inputs: Positions in [0, 1] range
        
        Returns:
            Encoded features
        """
        features_list = []
        
        for level in range(self.num_levels):
            resolution = int(self.base_resolution * (self.b ** level))
            
            # Compute grid coordinates
            grid_coords = inputs * resolution
            grid_coords_int = tf.cast(tf.floor(grid_coords), tf.int32)
            
            # Hash the integer coordinates
            hashed_idx = self._hash_coords(grid_coords_int, level)
            
            # Retrieve features from hash table
            features = tf.gather(self.hash_tables[level], hashed_idx)
            
            # Trilinear interpolation for smoother features
            local_coords = grid_coords - tf.cast(grid_coords_int, tf.float32)
            features = self._trilinear_interpolation(features, local_coords)
            
            features_list.append(features)
        
        return tf.concat(features_list, axis=-1)
    
    def _hash_coords(self, coords: tf.Tensor, level: int) -> tf.Tensor:
        """Hash integer coordinates to table indices."""
        primes = tf.constant([1, 2654435761, 805459861])
        table_size = 2 ** self.log2_hashmap_size
        
        # Simple spatial hashing
        hashed = coords[..., 0] * primes[0]
        hashed ^= coords[..., 1] * primes[1]
        hashed ^= coords[..., 2] * primes[2]
        
        return hashed % table_size
    
    def _trilinear_interpolation(self, features: tf.Tensor, 
                                 local_coords: tf.Tensor) -> tf.Tensor:
        """Apply trilinear interpolation for smooth features."""
        # Simplified version - full implementation would handle 8 corners
        return features * (1.0 - tf.reduce_mean(local_coords, axis=-1, keepdims=True))


def get_encoder(encoding_type: str = 'cheaper_nerf', **kwargs) -> Tuple[Callable, int]:
    """
    Factory function to get the appropriate encoder.
    
    Args:
        encoding_type: Type of encoding ('nerf', 'cheaper_nerf', 'hash')
        **kwargs: Additional arguments for the encoder
    
    Returns:
        Tuple of (encoding function, output dimension)
    """
    if encoding_type == 'nerf':
        config = EncodingConfig(**kwargs)
        encoder = PositionalEncoder(config)
    elif encoding_type == 'cheaper_nerf':
        config = EncodingConfig(**kwargs)
        encoder = CheaperPositionalEncoder(config)
    elif encoding_type == 'integrated':
        config = EncodingConfig(**kwargs)
        encoder = IntegratedPositionalEncoder(config)
    elif encoding_type == 'hash':
        encoder = HashEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    return encoder.encode, encoder.out_dim


# Convenience functions for common use cases
def get_embedder(multires: int, i: int = 0, 
                 encoding_type: str = 'cheaper_nerf') -> Tuple[Callable, int]:
    """
    Get embedder for spatial coordinates.
    
    Args:
        multires: log2 of max frequency
        i: Set to -1 for identity encoding
        encoding_type: Type of encoding to use
    
    Returns:
        Tuple of (encoding function, output dimension)
    """
    if i == -1:
        return lambda x: x, 3
    
    return get_encoder(
        encoding_type=encoding_type,
        input_dims=3,
        max_freq_log2=multires-1,
        num_freqs=multires,
        log_sampling=True
    )


def get_embedder_view(multires_views: int, 
                     encoding_type: str = 'cheaper_nerf') -> Tuple[Callable, int]:
    """
    Get embedder for viewing directions.
    
    Args:
        multires_views: log2 of max frequency for views
        encoding_type: Type of encoding to use
    
    Returns:
        Tuple of (encoding function, output dimension)
    """
    return get_encoder(
        encoding_type=encoding_type,
        input_dims=3,
        max_freq_log2=multires_views-1,
        num_freqs=multires_views,
        log_sampling=True
    )
