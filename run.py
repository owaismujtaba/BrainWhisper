#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent / "src"))

from whisper_eeg.cli.train import train
from whisper_eeg.cli.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="EEG-to-Text Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    inference_parser.add_argument("--input", type=str, default=None, help="Path to HDF5 file (optional, defaults to test set)")
    inference_parser.add_argument("--output", type=str, default="results/predictions.csv", help="Output file")
    inference_parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to use (default: val)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args.config)
    elif args.command == "inference":
        run_inference(args.checkpoint, args.input, args.output, args.split)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
