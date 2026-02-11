# BrainWhisper

**EEG-to-Text Decoding via Hallucinated Audio Modality**

A modular implementation of the "Hallucinating a Modality" approach for converting EEG signals to text using synthetic audio as an intermediary representation.

## Quick Start

### 1. Generate Synthetic Audio
```bash
.venv/bin/python run.py --mode generate_audio --config config.yaml
```

### 2. Train EEG Encoder
```bash
.venv/bin/python run.py --mode train --config config.yaml
```

## Project Structure

```
BrainWhisper/
├── run.py                          # Main entry point
├── config.yaml                     # Configuration file
├── src/brainwhisper/              # Main package
│   ├── models/                    # Model architectures
│   │   └── eeg_encoder.py        # EEG encoder (CNN + Transformer)
│   ├── data/                      # Data handling
│   │   ├── dataset.py            # EEG-Audio dataset
│   │   └── audio_generator.py    # TTS audio generation
│   ├── training/                  # Training logic
│   │   └── trainer.py            # Teacher-Student trainer
│   └── utils/                     # Utilities
│       └── config.py             # Config management
├── checkpoints/                   # Model checkpoints
└── data/                         # Data directory
    ├── raw/hdf5_data_final/      # EEG data (HDF5)
    └── audio_ground_truth/       # Generated audio
```

## Configuration

All parameters are in `config.yaml`:

- **Paths**: Data directories, checkpoint location
- **Audio**: TTS model, GPU settings
- **Model**: Architecture parameters (channels, layers, etc.)
- **Training**: Batch size, epochs, learning rate
- **Data**: Train/val splits, optional limit for debugging

## Architecture

### EEG Encoder
- **Input**: EEG features (Time, 512)
- **CNN Backbone**: 4 layers of 1D convolutions
- **Temporal Modeling**: 2-layer Transformer Encoder
- **Adapter**: Linear projection to Whisper embedding space
- **Output**: Embeddings (Batch, Time, 512)

### Training Strategy
1. **Teacher (Frozen)**: Whisper encoder processes synthetic audio
2. **Student (Trainable)**: EEG encoder learns to match teacher embeddings
3. **Loss**: MSE between student and teacher outputs

## Environment

- **Python**: 3.10 (via `uv` virtual environment)
- **Key Dependencies**:
  - `torch==2.10.0`
  - `openai-whisper==20250625`
  - `TTS==0.22.0`
  - `h5py`, `soundfile`, `pyyaml`

## Usage Examples

### Test Run (4 samples, 1 epoch)
```bash
.venv/bin/python run.py --mode train --config config_test.yaml
```

### Full Training (50 epochs)
Edit `config.yaml`:
```yaml
training:
  epochs: 50
  batch_size: 8
data:
  limit: null  # Use full dataset
```

Then run:
```bash
.venv/bin/python run.py --mode train --config config.yaml
```

## Checkpoints

Saved to `checkpoints/`:
- `best_eeg_encoder.pth` - Best validation loss
- `last_eeg_encoder.pth` - Latest epoch
- `checkpoint_epoch_N.pth` - Periodic saves

## Notes

- Audio generation uses Coqui TTS (no ffmpeg dependency via soundfile)
- Variable-length EEG sequences handled via padding
- Automatic temporal alignment between student and teacher outputs
