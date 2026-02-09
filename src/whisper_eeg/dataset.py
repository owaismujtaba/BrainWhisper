"""
Dataset class for EEG-to-Text conversion
"""
import os
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob
from typing import Dict, List, Optional, Tuple, Any
from transformers import WhisperTokenizer

class EEGTextDataset(Dataset):
    """
    Dataset for loading EEG and text transcriptions from HDF5 files.
    """
    
    def __init__(
        self,
        hdf5_dir: str,
        tokenizer: Any = None,
        max_eeg_length: int = 2000,
        max_text_length: int = 448,
        pad_eeg: bool = True,
        gaussian_noise: float = 0.0,
    ):
        """
        Initialize the dataset.
        
        Args:
            hdf5_dir: Directory containing HDF5 files
            tokenizer: HuggingFace tokenizer (optional, converts text to IDs)
            max_eeg_length: Maximum EEG sequence length
            max_text_length: Maximum text sequence length
            gaussian_noise: Standard deviation of Gaussian noise to add (augmentation)
        """
        self.hdf5_dir = hdf5_dir
        self.tokenizer = tokenizer
        self.max_eeg_length = max_eeg_length
        self.max_text_length = max_text_length
        self.pad_eeg = pad_eeg
        self.gaussian_noise = gaussian_noise
        
        # Find all HDF5 files in directory
        self.hdf5_files = sorted(Path(hdf5_dir).glob('**/data_*.hdf5'))
        
        # Map trial indices to files and trial names
        self.trial_map = []
        for hdf5_file in self.hdf5_files:
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    for trial_name in f.keys():
                        self.trial_map.append((str(hdf5_file), trial_name))
            except Exception as e:
                print(f"Error reading {hdf5_file}: {e}")
        
        print(f"Found {len(self.hdf5_files)} HDF5 files with {len(self.trial_map)} total trials")
    
    def __len__(self):
        return len(self.trial_map)
    
    def __getitem__(self, idx):
        hdf5_file, trial_name = self.trial_map[idx]
        
        with h5py.File(hdf5_file, 'r') as f:
            trial = f[trial_name]
            
            # Load EEG data: e.g. (773, 512)
            eeg = trial['input_features'][()].astype(np.float32)
            
            # Load text: (500,) ASCII codes
            text_codes = trial['transcription'][()].astype(np.int64)
            text_raw = self._codes_to_text(text_codes)
            
            # Process EEG (Trim/Pad/Augment)
            eeg = self._process_eeg(eeg)
            
            # Process Text (Tokenize for Whisper)
            if self.tokenizer:
                # Tokenize (Force English)
                # Whisper tokenizer sets language prefix if we validly specify it, 
                # or we just rely on the inputs being English. 
                # Ideally, we should set the decoder start token to English.
                # Standard tokenizer encode usually adds <|startoftranscript|><|en|><|transcribe|>...
                
                # Check tokenizer capabilities
                # For basic HuggingFace WhisperTokenizer, we can't always pass language to __call__ directly 
                # in the same way as processor. But we can prepending tokens manually or rely on model defaults.
                # Actually, standard tokenizer just encodes text. The model's `forced_decoder_ids` handles the prefix during generation.
                # BUT for training, the `labels` should ideally start with the correct start tokens if we want the model to learn them.
                # However, usually we just give the text tokens and let the model's forward pass handle the shifting.
                # WhisperForConditionalGeneration handles shifting labels to decoder_input_ids automatically.
                
                labels = self.tokenizer(
                    text_raw, 
                    max_length=self.max_text_length, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.squeeze(0) # (max_text_length,)
                
                # Mask padding tokens with -100 so loss is not computed on them
                labels[labels == self.tokenizer.pad_token_id] = -100
                
            else:
                # Legacy behavior (not used with Whisper)
                labels = torch.from_numpy(self._process_text_legacy(text_codes))
        
        return {
            'eeg': torch.from_numpy(eeg),
            'labels': labels, 
            'text_raw': text_raw # Useful for debug/WER calculation
        }
    
    def _codes_to_text(self, codes):
        """Convert custom ASCII codes back to string. Filter out 0 (padding)."""
        valid_codes = codes[codes != 0]
        # Filter mostly ASCII range to avoid errors
        return ''.join(chr(int(c)) for c in valid_codes if 0 < c < 256)
    
    def _process_eeg(self, eeg):
        """Process EEG sequence: trim, pad, and augment."""
        # Trim if too long
        if eeg.shape[0] > self.max_eeg_length:
            eeg = eeg[:self.max_eeg_length]
        
        # Pad if requested
        if self.pad_eeg and eeg.shape[0] < self.max_eeg_length:
            pad_width = ((0, self.max_eeg_length - eeg.shape[0]), (0, 0))
            eeg = np.pad(eeg, pad_width, mode='constant', constant_values=0)
            
        # Data Augmentation: Gaussian Noise (Only during training if set)
        if self.gaussian_noise > 0:
            noise = np.random.normal(0, self.gaussian_noise, eeg.shape).astype(np.float32)
            eeg = eeg + noise
            
        # Normalization (Standard Scaling per trial)
        mean = eeg.mean(axis=0, keepdims=True)
        std = eeg.std(axis=0, keepdims=True)
        eeg = (eeg - mean) / (std + 1e-6)
            
        return eeg
    
    def _process_text_legacy(self, text):
        """Legacy text processing."""
        if text.shape[0] > self.max_text_length:
            text = text[:self.max_text_length]
        if text.shape[0] < self.max_text_length:
            text = np.pad(text, (0, self.max_text_length - text.shape[0]),
                         mode='constant', constant_values=0)
        return text


def create_dataloaders(
    data_dir: str,
    tokenizer: Any = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_eeg_length: int = 2000,
    max_text_length: int = 448,
    gaussian_noise: float = 0.0,
) -> Dict[str, Optional[DataLoader]]:
    
    def collate_fn(batch):
        eeg_batch = torch.stack([item['eeg'] for item in batch])
        labels_batch = torch.stack([item['labels'] for item in batch])
        
        # EEG Mask (1 for real data, 0 for padding)
        # Assuming padding introduced 0s in all channels
        eeg_mask = (eeg_batch.sum(dim=2) != 0).float()
        
        return {
            'eeg': eeg_batch,
            'labels': labels_batch,
            'eeg_mask': eeg_mask,
            'text_raw': [item['text_raw'] for item in batch]
        }
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, '*', f'data_{split}.hdf5')
        matching_files = sorted(glob(split_path))
        
        if matching_files:
            # Apply augmentation only to TRAIN set
            noise = gaussian_noise if split == 'train' else 0.0
            
            dataset = EEGTextDataset(
                hdf5_dir=data_dir,
                tokenizer=tokenizer,
                max_eeg_length=max_eeg_length,
                max_text_length=max_text_length,
                pad_eeg=True,
                gaussian_noise=noise, 
            )
            
            # Filter
            filtered_indices = []
            for i, (hdf5_file, _) in enumerate(dataset.trial_map):
                if f'data_{split}.hdf5' in hdf5_file:
                    filtered_indices.append(i)
            dataset.trial_map = [dataset.trial_map[i] for i in filtered_indices]
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            print(f"{split.upper()} dataset: {len(dataset)} samples")
        else:
            loaders[split] = None
    
    return loaders
