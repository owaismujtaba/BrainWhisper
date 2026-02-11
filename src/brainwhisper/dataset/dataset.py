"""Dataset classes for EEG-Audio paired data"""

import os
import torch
import h5py
import numpy as np
import whisper
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Dataset for paired EEG and Audio data"""
    
    def __init__(self, data_dir, audio_dir, split="train", limit=None):
        """
        Args:
            data_dir: Directory containing HDF5 files
            audio_dir: Directory containing generated audio files
            split: Data split ('train' or 'val')
            limit: Optional limit on number of samples (for debugging)
        """
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.split = split
        self.samples = []
        
        print(f"Scanning dataset for split '{split}'...")
        
        # Find HDF5 files
        search_path = os.path.join(data_dir, split) if os.path.exists(os.path.join(data_dir, split)) else data_dir
        pattern = f"data_{split}.hdf5"
        hdf5_files = sorted(Path(search_path).rglob(pattern))
        
        if not hdf5_files:
            # Try generic pattern if split specific not found
            hdf5_files = sorted(Path(search_path).rglob("*.hdf5"))

        for h5_path in tqdm(hdf5_files, desc="Indexing HDF5 files"):
            try:
                # Get subject name from parent folder
                subject_name = h5_path.parent.name
                
                with h5py.File(h5_path, 'r') as f:
                    for trial_name in f.keys():
                        # Check if corresponding audio exists
                        safe_name = "".join(c for c in trial_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(" ", "_")
                        audio_path = os.path.join(audio_dir, split, subject_name, f"{safe_name}.wav")
                        
                        if os.path.exists(audio_path):
                            self.samples.append((str(h5_path), trial_name, audio_path))
            except Exception as e:
                print(f"Error reading {h5_path}: {e}")
                
        if limit:
            self.samples = self.samples[:limit]
            
        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, trial_name, audio_path = self.samples[idx]
        
        # Load EEG
        with h5py.File(h5_path, 'r') as f:
            trial = f[trial_name]
            # Using 'input_features' as identified in HDF5 inspection
            if 'input_features' in trial:
                eeg_data = trial['input_features'][()]
            elif 'raw' in trial:
                eeg_data = trial['raw'][()]
            elif 'data' in trial:
                eeg_data = trial['data'][()]
            else:
                # Last resort: try to find a dataset with shape
                for k in trial.keys():
                    if isinstance(trial[k], h5py.Dataset) and len(trial[k].shape) >= 2:
                        eeg_data = trial[k][()]
                        break
        
        # input_features is already in (Time, Channels) format: (910, 512)
        # No need to transpose
        
        # Convert to float32 tensor
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Load Audio and compute Mel Spectrogram
        # Whisper expects 30s padded audio @ 16kHz
        audio_np, sr = sf.read(audio_path)
        
        # Convert to tensor (Time, Channels) or (Time) -> (Channels, Time)
        audio = torch.from_numpy(audio_np).float()
        
        # soundfile returns (Time, Channels) if stereo, or (Time) if mono
        if len(audio.shape) > 1:
            audio = audio.transpose(0, 1)  # (Channels, Time)
        else:
            audio = audio.unsqueeze(0)  # (1, Time)

        # Resample to 16000 Hz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
            
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
            
        # Whisper expects numpy array or tensor
        # pad_or_trim expects 1D array/tensor
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        
        return eeg_tensor, mel


def collate_fn(batch):
    """Custom collate function to handle variable-length EEG sequences"""
    eegs, mels = zip(*batch)
    
    # EEG shape is (Time, Channels) from input_features
    # Pad EEG to max length in batch along time dimension (dim=0)
    max_len = max([e.shape[0] for e in eegs])
    padded_eegs = []
    for e in eegs:
        pad_len = max_len - e.shape[0]
        if pad_len > 0:
            # Pad along dim=0 (time): (pad_left, pad_right) for last dim, then second-to-last, etc.
            # For shape (Time, Channels), we want to pad time, so pad=(0, 0, 0, pad_len)
            # But F.pad works from last dim backwards, so for (Time, Channels):
            # pad=(0, 0) pads Channels, pad=(0, pad_len) pads Time
            e = torch.nn.functional.pad(e, (0, 0, 0, pad_len))
        padded_eegs.append(e)
        
    eeg_batch = torch.stack(padded_eegs)
    mel_batch = torch.stack(mels)
    
    return eeg_batch, mel_batch
