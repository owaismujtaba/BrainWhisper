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
from brainwhisper.data import AudioGenerator
from brainwhisper.training import Trainer


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


def main():
    parser = argparse.ArgumentParser(description="BrainWhisper: EEG-to-Text")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate_audio", "train"],
        required=True,
        help="Mode to run: generate_audio or train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
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


if __name__ == "__main__":
    main()
