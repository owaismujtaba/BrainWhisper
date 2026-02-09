# BrainWhisper: EEG-to-Text Model

BrainWhisper is a deep learning project designed to translate EEG (Electroencephalography) brain signals directly into text. It leverages the power of **OpenAI's Whisper** model, adapting its robust decoder to understand neural representations of speech.

## 🚀 Features

- **EEG Encoder**: A custom transformer-based encoder designed for EEG signals.
  - **Multi-Scale Convolution**: Captures features at different frequency bands (Delta, Theta, Alpha, Beta, Gamma).
  - **Gated Adapter**: Aligns EEG feature distributions with the text latent space.
  - **Adaptive Normalization**: Ensures statistical compatibility with the Whisper decoder.
- **Whisper Integration**: Uses pre-trained OpenAI Whisper (e.g., `openai/whisper-tiny`) as the text decoder.
- **Hybrid Training**:
  - **Cross-Attention Tuning**: Unfreezes specific layers to adapt Whisper to EEG inputs.
  - **CTC Auxiliary Loss**: Uses Connectionist Temporal Classification (CTC) to align phonetic features during training.
- **Data Augmentation**: Gaussian noise injection for robust training.

## 🧠 Architecture

The model consists of two main components:

1.  **EEG Encoder**: 
    - Takes raw EEG features (e.g., Mel-spectrograms or raw signals) as input.
    - Applies multi-scale convolutions for temporal downsampling.
    - Uses a Transformer Encoder to capture long-range dependencies.
    - Projects features to `d_model` dimension of the specific Whisper model.

2.  **Whisper Decoder**:
    - Generates text tokens autoregressively.
    - Cross-attention layers attend to the output of the EEG Encoder instead of Mel-spectrograms.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/BrainWhisper.git
    cd BrainWhisper
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

The model expects data in **HDF5 format**.
- **Directory Structure**:
  ```text
  data/
  └── raw/
      └── hdf5_data_final/
          ├── train/data_train.hdf5
          ├── val/data_val.hdf5
          └── test/data_test.hdf5
  ```
- **HDF5 Content**:
  Each file should contain groups for trials, with datasets:
  - `input_features`: EEG signal matrix (Shape: `[Time, Channels]`, e.g., `[2000, 512]`)
  - `transcription`: ASCII codes of the target text (Shape: `[Length]`)

## ⚙️ Configuration

Hyperparameters are managed in `configs/default.yaml`. Key settings include:

```yaml
data:
  max_eeg_length: 2000      # Max time steps for EEG
  gaussian_noise: 0.1       # Augmentation strength

model:
  whisper_model_name: "openai/whisper-tiny" # Base model
  freeze_decoder: true      # Freeze Whisper weights (except cross-attn)
  eeg_input_dim: 512        # Input EEG feature dimension
  encoder_layers: 6         # Depth of EEG Encoder

training:
  batch_size: 80
  learning_rate: 1.0e-4
  ctc_weight: 0.3           # Weight for auxiliary CTC loss
```

## 🏃 Usage

### 1. Training

To train the model using the default configuration:

```bash
./run.py train --config configs/default.yaml
```

### 2. Inference

To generate text from EEG data using a trained checkpoint:

```bash
./run.py inference --checkpoint checkpoints/best_model.pt --input data/test/data_test.hdf5 --output results.csv
```

### 3. Testing Pipeline

To verify that the installation and pipeline are working correctly (runs a forward pass on a sample):

```bash
python test_pipeline.py
```

## 📁 Project Structure

```text
.
├── run.py                  # CLI Entry point
├── test_pipeline.py        # Sanity check script
├── configs/                # Configuration files
├── data/                   # Data directory
├── src/
│   └── whisper_eeg/
│       ├── dataset.py      # HDF5 Data loading
│       ├── model.py        # EEGEncoder & EEGWhisperModel
│       ├── trainer.py      # Training loop & Validation
│       ├── config.py       # Config dataclasses
│       └── cli/            # CLI command implementations
└── requirements.txt        # Python dependencies
```

## 📜 License

[MIT License](LICENSE)
