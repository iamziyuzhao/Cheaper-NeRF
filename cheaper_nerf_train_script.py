"""
train.py - Main Training Script for Cheaper-NeRF
Entry point for training Cheaper-NeRF models.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cheaper_nerf.trainer import TrainingConfig, create_trainer
from cheaper_nerf.data_loader import load_nerf_data
from cheaper_nerf.utils import set_random_seed, get_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Cheaper-NeRF: Cost-Efficient Novel-View Synthesis'
    )
    
    # Experiment
    parser.add_argument('--config', type=str, 
                       help='Config file path')
    parser.add_argument('--expname', type=str, default='cheaper_nerf',
                       help='Experiment name')
    parser.add_argument('--basedir', type=str, default='./logs',
                       help='Where to store checkpoints and logs')
    
    # Data
    parser.add_argument('--datadir', type=str, required=True,
                       help='Input data directory')
    parser.add_argument('--dataset_type', type=str, default='blender',
                       choices=['blender', 'llff', 'deepvoxels'],
                       help='Dataset type')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='cheaper_nerf',
                       choices=['cheaper_nerf', 'lightweight', 'adaptive'],
                       help='Model architecture type')
    parser.add_argument('--netdepth', type=int, default=6,
                       help='Network depth (layers)')
    parser.add_argument('--netwidth', type=int, default=128,
                       help='Network width (channels)')
    parser.add_argument('--netdepth_fine', type=int, default=6,
                       help='Fine network depth')
    parser.add_argument('--netwidth_fine', type=int, default=128,
                       help='Fine network width')
    
    # Sampling
    parser.add_argument('--N_samples', type=int, default=32,
                       help='Number of coarse samples (reduced from 64)')
    parser.add_argument('--N_importance', type=int, default=64,
                       help='Number of fine samples (reduced from 128)')
    parser.add_argument('--perturb', type=float, default=1.,
                       help='Perturbation for sampling')
    
    # Cheaper-NeRF specific
    parser.add_argument('--enable_mean_sampling', action='store_true', default=True,
                       help='Enable mean sampling reduction')
    parser.add_argument('--sampling_reduction', type=int, default=4,
                       help='Factor for mean sampling reduction')
    parser.add_argument('--enable_zero_filtering', action='store_true', default=True,
                       help='Enable zero-density filtering')
    parser.add_argument('--density_threshold', type=float, default=1e-10,
                       help='Threshold for density filtering')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size (number of rays)')
    parser.add_argument('--lrate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--lrate_decay', type=int, default=250,
                       help='Learning rate decay (in 1000s)')
    parser.add_argument('--N_iters', type=int, default=200000,
                       help='Number of training iterations')
    
    # Rendering
    parser.add_argument('--use_viewdirs', action='store_true', default=True,
                       help='Use view directions')
    parser.add_argument('--white_bkgd', action='store_true',
                       help='White background for synthetic data')
    parser.add_argument('--raw_noise_std', type=float, default=0.,
                       help='Noise for density regularization')
    
    # Positional encoding
    parser.add_argument('--multires', type=int, default=10,
                       help='Log2 of max frequency for position encoding')
    parser.add_argument('--multires_views', type=int, default=4,
                       help='Log2 of max frequency for direction encoding')
    
    # Logging
    parser.add_argument('--i_print', type=int, default=100,
                       help='Print frequency')
    parser.add_argument('--i_img', type=int, default=500,
                       help='Image logging frequency')
    parser.add_argument('--i_weights', type=int, default=5000,
                       help='Checkpoint frequency')
    parser.add_argument('--i_testset', type=int, default=2500,
                       help='Test set evaluation frequency')
    parser.add_argument('--i_video', type=int, default=10000,
                       help='Video generation frequency')
    
    # System
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--benchmark_mode', action='store_true',
                       help='Run in benchmark mode for performance testing')
    
    return parser.parse_args()


def setup_gpu(gpu_id: int):
    """Setup GPU configuration."""
    # Set visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU {gpu_id}: {gpus[0].name}")
            
            # Print available memory
            memory_info = get_gpu_memory()
            print(f"Available GPU memory: {memory_info:.2f} GB")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs available, using CPU")


def load_config_file(config_path: str) -> dict:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith('.yaml'):
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def print_banner():
    """Print Cheaper-NeRF banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        Cheaper-NeRF: Cost-Efficient Novel-View          ║
    ║                     Synthesis                           ║
    ║                                                          ║
    ║    ~40% faster training | ~60% memory reduction         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(config: TrainingConfig):
    """Print configuration summary."""
    print("\nConfiguration Summary:")
    print("-" * 50)
    print(f"Experiment: {config.expname}")
    print(f"Model Type: {config.model_type}")
    print(f"Network: {config.netdepth} layers × {config.netwidth} channels")
    print(f"Sampling: {config.N_samples} coarse + {config.N_importance} fine")
    print(f"Batch Size: {config.batch_size} rays")
    print(f"Iterations: {config.N_iters}")
    print("\nCheaper-NeRF Optimizations:")
    print(f"  Mean Sampling: {'✓' if config.enable_mean_sampling else '✗'} (factor: {config.sampling_reduction})")
    print(f"  Zero Filtering: {'✓' if config.enable_zero_filtering else '✗'} (threshold: {config.density_threshold})")
    print("-" * 50)


def benchmark_mode(config: TrainingConfig):
    """Run in benchmark mode for performance comparison."""
    print("\n" + "="*60)
    print("BENCHMARK MODE")
    print("="*60)
    
    # Test with optimizations
    print("\n1. Testing WITH Cheaper-NeRF optimizations...")
    config.enable_mean_sampling = True
    config.enable_zero_filtering = True
    trainer_optimized = create_trainer(config.__dict__)
    
    # Dummy data for benchmarking
    dummy_data = {
        'rays': np.random.randn(10000, 11).astype(np.float32),
        'rgb': np.random.rand(10000, 3).astype(np.float32)
    }
    
    import time
    start = time.time()
    for _ in range(100):
        trainer_optimized.train_step(
            tf.constant(dummy_data['rays'][:config.batch_size, :3]),
            tf.constant(dummy_data['rays'][:config.batch_size, 3:6]),
            tf.constant(dummy_data['rays'][:config.batch_size, 8:11]),
            tf.constant(dummy_data['rgb'][:config.batch_size]),
            tf.constant(dummy_data['rays'][:config.batch_size, 6:7]),
            tf.constant(dummy_data['rays'][:config.batch_size, 7:8])
        )
    time_optimized = time.time() - start
    
    # Test without optimizations
    print("\n2. Testing WITHOUT optimizations...")
    config.enable_mean_sampling = False
    config.enable_zero_filtering = False
    config.N_samples = 64  # Original values
    config.N_importance = 128
    trainer_baseline = create_trainer(config.__dict__)
    
    start = time.time()
    for _ in range(100):
        trainer_baseline.train_step(
            tf.constant(dummy_data['rays'][:config.batch_size, :3]),
            tf.constant(dummy_data['rays'][:config.batch_size, 3:6]),
            tf.constant(dummy_data['rays'][:config.batch_size, 8:11]),
            tf.constant(dummy_data['rgb'][:config.batch_size]),
            tf.constant(dummy_data['rays'][:config.batch_size, 6:7]),
            tf.constant(dummy_data['rays'][:config.batch_size, 7:8])
        )
    time_baseline = time.time() - start
    
    # Results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Baseline NeRF: {time_baseline:.2f} seconds")
    print(f"Cheaper-NeRF:  {time_optimized:.2f} seconds")
    print(f"Speedup: {time_baseline/time_optimized:.2f}x")
    print(f"Time Saved: {((time_baseline-time_optimized)/time_baseline)*100:.1f}%")
    print("="*60)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Setup GPU
    setup_gpu(args.gpu)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create configuration
    if args.config:
        # Load from config file
        config_dict = load_config_file(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
    else:
        # Use command line arguments
        config_dict = vars(args)
    
    config = TrainingConfig(**config_dict)
    
    # Print configuration
    print_config(config)
    
    # Run benchmark mode if requested
    if args.benchmark_mode:
        benchmark_mode(config)
        return
    
    # Load data
    print("\nLoading dataset...")
    train_data, val_data, test_data = load_nerf_data(
        args.datadir,
        args.dataset_type,
        args.white_bkgd
    )
    print(f"Loaded {len(train_data['images'])} training images")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer(config_dict)
    
    # Start training
    print("\nStarting training...")
    try:
        trainer.train(train_data, val_data, test_data)
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        save_checkpoint = input("Save checkpoint? (y/n): ")
        if save_checkpoint.lower() == 'y':
            trainer.save_checkpoint(trainer.global_step)
            print("Checkpoint saved")
    
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
