"""
network.py - Neural Network Architectures for Cheaper-NeRF
Implements optimized MLP architectures with reduced parameters.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, List, Tuple
import numpy as np


class CheaperNeRFModel(keras.Model):
    """
    Optimized NeRF model with reduced parameters for faster training.
    Key optimizations:
    - Reduced depth (6 vs 8 layers)
    - Reduced width (128 vs 256 channels)
    - Efficient skip connections
    """
    
    def __init__(self,
                 D: int = 6,              # Reduced from 8
                 W: int = 128,            # Reduced from 256
                 input_ch: int = 63,      # Position encoding channels
                 input_ch_views: int = 27, # View encoding channels
                 output_ch: int = 4,      # RGB + density
                 skips: List[int] = [3],  # Skip connection at layer 3
                 use_viewdirs: bool = True,
                 activation: str = 'relu',
                 output_activation: str = 'none'):
        super(CheaperNeRFModel, self).__init__()
        
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Choose activation function
        self.activation = self._get_activation(activation)
        
        # Build network layers
        self._build_layers()
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'relu': tf.nn.relu,
            'leaky_relu': tf.nn.leaky_relu,
            'elu': tf.nn.elu,
            'selu': tf.nn.selu,
            'gelu': tf.nn.gelu
        }
        return activations.get(name, tf.nn.relu)
    
    def _build_layers(self):
        """Build the network layers."""
        # Position encoding layers
        self.pts_linears = []
        for i in range(self.D):
            if i == 0:
                input_dim = self.input_ch
            elif i in self.skips:
                input_dim = self.W + self.input_ch
            else:
                input_dim = self.W
            
            layer = keras.layers.Dense(self.W, activation=None)
            self.pts_linears.append(layer)
        
        # View direction encoding layers
        if self.use_viewdirs:
            self.views_linears = []
            # Feature vector layer
            self.feature_linear = keras.layers.Dense(self.W, activation=None)
            
            # View-dependent layers (smaller network)
            self.views_linears.append(
                keras.layers.Dense(self.W // 2, activation=None)
            )
            
            # Alpha (density) output - view independent
            self.alpha_linear = keras.layers.Dense(1, activation=None)
            
            # RGB output - view dependent
            self.rgb_linear = keras.layers.Dense(3, activation=None)
        else:
            # Single output layer for both RGB and density
            self.output_linear = keras.layers.Dense(4, activation=None)
        
        # Batch normalization for stability (optional)
        self.use_batch_norm = False
        if self.use_batch_norm:
            self.batch_norms = [
                keras.layers.BatchNormalization() for _ in range(self.D)
            ]
    
    def call(self, inputs, training=False):
        """
        Forward pass through the network.
        
        Args:
            inputs: Either [pts] or [pts, views] depending on use_viewdirs
            training: Whether in training mode
        
        Returns:
            output: [RGB, sigma] tensor of shape [..., 4]
        """
        if self.use_viewdirs:
            input_pts, input_views = inputs
        else:
            input_pts = inputs
        
        # Pass through position encoding layers
        h = input_pts
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = self.activation(h)
            
            # Apply skip connection
            if i in self.skips:
                h = tf.concat([input_pts, h], -1)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and i < len(self.batch_norms):
                h = self.batch_norms[i](h, training=training)
        
        if self.use_viewdirs:
            # Output density (view-independent)
            alpha = self.alpha_linear(h)
            
            # Extract feature vector
            feature = self.feature_linear(h)
            
            # Concatenate with view direction
            h = tf.concat([feature, input_views], -1)
            
            # Pass through view-dependent layers
            for layer in self.views_linears:
                h = layer(h)
                h = self.activation(h)
            
            # Output RGB
            rgb = self.rgb_linear(h)
            
            # Combine outputs
            outputs = tf.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs


class LightweightNeRF(keras.Model):
    """
    Ultra-lightweight NeRF variant for real-time applications.
    Further reduced parameters for mobile/edge deployment.
    """
    
    def __init__(self,
                 D: int = 4,              # Very shallow
                 W: int = 64,             # Narrow network
                 input_ch: int = 63,
                 input_ch_views: int = 27,
                 use_viewdirs: bool = False,  # Disable for speed
                 dropout_rate: float = 0.0):
        super(LightweightNeRF, self).__init__()
        
        self.D = D
        self.W = W
        self.use_viewdirs = use_viewdirs
        self.dropout_rate = dropout_rate
        
        # Build compact network
        self.layers = []
        input_dim = input_ch
        
        for i in range(D):
            self.layers.append(
                keras.layers.Dense(W if i < D-1 else 4, activation='relu')
            )
            
            if dropout_rate > 0 and i < D-1:
                self.layers.append(
                    keras.layers.Dropout(dropout_rate)
                )
    
    def call(self, inputs, training=False):
        """Simplified forward pass."""
        h = inputs[0] if isinstance(inputs, list) else inputs
        
        for layer in self.layers:
            h = layer(h, training=training) if hasattr(layer, 'rate') else layer(h)
        
        return h


class CoarseToFineNeRF(keras.Model):
    """
    Hierarchical NeRF with separate coarse and fine networks.
    Implements the two-stage sampling strategy of Cheaper-NeRF.
    """
    
    def __init__(self,
                 coarse_config: dict,
                 fine_config: dict = None,
                 share_weights: bool = False):
        super(CoarseToFineNeRF, self).__init__()
        
        # Create coarse network
        self.coarse_model = CheaperNeRFModel(**coarse_config)
        
        # Create fine network (optionally share weights)
        if share_weights:
            self.fine_model = self.coarse_model
        elif fine_config is not None:
            self.fine_model = CheaperNeRFModel(**fine_config)
        else:
            self.fine_model = None
        
        self.share_weights = share_weights
    
    def call_coarse(self, inputs, training=False):
        """Forward pass through coarse network."""
        return self.coarse_model(inputs, training=training)
    
    def call_fine(self, inputs, training=False):
        """Forward pass through fine network."""
        if self.fine_model is not None:
            return self.fine_model(inputs, training=training)
        return self.call_coarse(inputs, training=training)
    
    def call(self, inputs, stage='both', training=False):
        """
        Forward pass through the hierarchical network.
        
        Args:
            inputs: Network inputs
            stage: 'coarse', 'fine', or 'both'
            training: Whether in training mode
        
        Returns:
            Output from specified stage(s)
        """
        if stage == 'coarse':
            return self.call_coarse(inputs, training=training)
        elif stage == 'fine':
            return self.call_fine(inputs, training=training)
        else:
            # Return both for hierarchical sampling
            coarse_output = self.call_coarse(inputs, training=training)
            fine_output = self.call_fine(inputs, training=training)
            return coarse_output, fine_output


class AdaptiveNeRF(CheaperNeRFModel):
    """
    Adaptive NeRF that adjusts capacity based on scene complexity.
    Dynamically enables/disables layers during inference.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_threshold = 0.5
        self.layer_importance = None
        
    def estimate_complexity(self, inputs):
        """Estimate scene complexity from input distribution."""
        if isinstance(inputs, list):
            pts = inputs[0]
        else:
            pts = inputs
        
        # Simple complexity metric based on spatial variance
        spatial_variance = tf.reduce_mean(tf.math.reduce_variance(pts, axis=-2))
        return spatial_variance
    
    def call(self, inputs, training=False, adaptive=True):
        """Forward pass with optional adaptive computation."""
        if not adaptive or training:
            return super().call(inputs, training=training)
        
        # Estimate complexity
        complexity = self.estimate_complexity(inputs)
        
        # Adaptively skip layers for simple regions
        if complexity < self.complexity_threshold:
            # Use simplified forward pass
            return self._fast_forward(inputs)
        else:
            return super().call(inputs, training=training)
    
    def _fast_forward(self, inputs):
        """Simplified forward pass for low-complexity regions."""
        if self.use_viewdirs:
            input_pts, input_views = inputs
        else:
            input_pts = inputs
        
        # Use only first and last layers
        h = self.pts_linears[0](input_pts)
        h = self.activation(h)
        h = self.pts_linears[-1](h)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = tf.concat([h, input_views], -1)
            rgb = self.rgb_linear(feature)
            return tf.concat([rgb, alpha], -1)
        else:
            return self.output_linear(h)


class EfficientNeRFEnsemble(keras.Model):
    """
    Ensemble of multiple smaller NeRF models for improved quality.
    Cheaper than a single large model while maintaining quality.
    """
    
    def __init__(self,
                 num_models: int = 3,
                 base_config: dict = None,
                 aggregation: str = 'mean'):
        super().__init__()
        
        self.num_models = num_models
        self.aggregation = aggregation
        
        # Default config for ensemble members
        if base_config is None:
            base_config = {
                'D': 4,
                'W': 96,
                'skips': [2]
            }
        
        # Create ensemble members
        self.models = [
            CheaperNeRFModel(**base_config) for _ in range(num_models)
        ]
    
    def call(self, inputs, training=False):
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            output = model(inputs, training=training)
            outputs.append(output)
        
        # Aggregate predictions
        outputs = tf.stack(outputs, axis=0)
        
        if self.aggregation == 'mean':
            return tf.reduce_mean(outputs, axis=0)
        elif self.aggregation == 'median':
            return tfp.stats.percentile(outputs, 50.0, axis=0)
        elif self.aggregation == 'weighted':
            # Learn weights for each model
            weights = tf.nn.softmax(self.ensemble_weights)
            return tf.reduce_sum(outputs * weights[:, None, None], axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


def create_nerf_model(model_type: str = 'cheaper_nerf', **kwargs) -> keras.Model:
    """
    Factory function to create NeRF models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated NeRF model
    """
    models = {
        'cheaper_nerf': CheaperNeRFModel,
        'lightweight': LightweightNeRF,
        'hierarchical': CoarseToFineNeRF,
        'adaptive': AdaptiveNeRF,
        'ensemble': EfficientNeRFEnsemble
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


# Model configurations for different use cases
MODEL_CONFIGS = {
    'mobile': {
        'D': 4,
        'W': 64,
        'skips': [],
        'use_viewdirs': False
    },
    'balanced': {
        'D': 6,
        'W': 128,
        'skips': [3],
        'use_viewdirs': True
    },
    'quality': {
        'D': 8,
        'W': 196,
        'skips': [4],
        'use_viewdirs': True
    },
    'original': {
        'D': 8,
        'W': 256,
        'skips': [4],
        'use_viewdirs': True
    }
}


def get_model_config(preset: str = 'balanced') -> dict:
    """Get predefined model configuration."""
    if preset not in MODEL_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}")
    return MODEL_CONFIGS[preset].copy()
