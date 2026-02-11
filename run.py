#!/usr/bin/env python3
"""
BrainWhisper: EEG-to-Text via Hallucinated Audio Modality

Main entry point for training and audio generation.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from brainwhisper.utils import load_config
from brainwhisper.dataset import AudioGenerator
from brainwhisper.training import Trainer
from brainwhisper.inference import InferencePipeline, print_evaluation_report


def generate_audio(config):
    """Generate synthetic audio from text transcriptions"""
    print("=" * 60)
    print("AUDIO GENERATION MODE")
    print("=" * 60)
    
    generator = AudioGenerator(
        tts_model=config.audio.tts_model,
        use_gpu=config.audio.use_gpu
    )
    
    # Generate for validation split
    print(f"\nGenerating audio for validation split...")
    generator.generate_from_hdf5(
        data_dir=config.paths.data_dir,
        output_dir=config.paths.audio_dir,
        split=config.data.val_split,
        limit=config.data.limit
    )
    
    # Generate for training split
    print(f"\nGenerating audio for training split...")
    generator.generate_from_hdf5(
        data_dir=config.paths.data_dir,
        output_dir=config.paths.audio_dir,
        split=config.data.train_split,
        limit=config.data.limit
    )
    
    print("\nAudio generation complete!")


def train_model(config):
    """Train EEG encoder"""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    trainer = Trainer(config)
    trainer.train()


def run_inference(config, checkpoint_path, hdf5_path, trial_name):
    """Run inference on EEG data"""
    print("=" * 60)
    print("INFERENCE MODE")
    print("=" * 60)
    
    pipeline = InferencePipeline(checkpoint_path, config)
    
    if hdf5_path and trial_name:
        # Single file inference
        print(f"\nProcessing: {hdf5_path} / {trial_name}")
        prediction = pipeline.predict_from_file(hdf5_path, trial_name)
        print(f"\nPrediction: {prediction}")
    else:
        # Batch inference on validation set
        print("\nRunning batch inference on validation set...")
        from pathlib import Path
        import h5py
        
        # Find all validation HDF5 files
        data_dir = Path(config.paths.data_dir)
        val_files = []
        
        for hdf5_file in data_dir.rglob("*val.hdf5"):
            with h5py.File(hdf5_file, 'r') as f:
                for trial in list(f.keys())[:5]:  # Limit to 5 per file for demo
                    val_files.append((str(hdf5_file), trial))
        
        print(f"Found {len(val_files)} validation trials")
        results = pipeline.batch_predict(val_files[:20])  # Process first 20
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for r in results:
            if r.get('error'):
                print(f"\n{r['trial']}: ERROR - {r['error']}")
            else:
                print(f"\n{r['trial']}")
                print(f"  Ground Truth: {r['ground_truth']}")
                print(f"  Prediction:   {r['prediction']}")
        
        # Print evaluation metrics
        print_evaluation_report(results)


def main():
    parser = argparse.ArgumentParser(description="BrainWhisper: EEG-to-Text")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate_audio", "train", "inference"],
        required=True,
        help="Mode to run: generate_audio, train, or inference",
        default="train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_eeg_encoder.pth",
        help="Path to model checkpoint (for inference)"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="Path to HDF5 file (for single file inference)"
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Trial name in HDF5 file (for single file inference)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Run selected mode
    if args.mode == "generate_audio":
        generate_audio(config)
    elif args.mode == "train":
        train_model(config)
    elif args.mode == "inference":
        run_inference(config, args.checkpoint, args.hdf5, args.trial)


if __name__ == "__main__":
    main()
