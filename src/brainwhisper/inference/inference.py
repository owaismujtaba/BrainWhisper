"""Inference pipeline for EEG-to-text generation"""

import torch
import whisper
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from ..models import EEGEncoder


class InferencePipeline:
    """Pipeline for generating text from EEG signals"""
    
    def __init__(self, checkpoint_path: str, config, device: str = "cuda"):
        """
        Initialize inference pipeline
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object
            device: Device to run inference on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading inference pipeline on {self.device}...")
        
        # Load Whisper model
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(config.model.whisper_model, device=self.device)
        
        # Load trained EEG encoder
        print(f"Loading EEG encoder from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        embed_dim = self.whisper_model.dims.n_audio_state
        self.eeg_encoder = EEGEncoder(
            input_channels=config.model.eeg_channels,
            output_dim=embed_dim,
            hidden_dim=config.model.hidden_dim,
            kernel_size=config.model.kernel_size,
            stride=config.model.stride,
            num_layers=config.model.num_cnn_layers,
            dropout=config.model.dropout
        ).to(self.device)
        
        self.eeg_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.eeg_encoder.eval()
        
        print("Inference pipeline ready!")
    
    def load_eeg_from_hdf5(self, hdf5_path: str, trial_name: str) -> torch.Tensor:
        """
        Load EEG data from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            trial_name: Name of trial to load
            
        Returns:
            EEG tensor of shape (Time, Channels)
        """
        with h5py.File(hdf5_path, 'r') as f:
            trial = f[trial_name]
            
            # Try different possible keys
            if 'input_features' in trial:
                eeg_data = trial['input_features'][()]
            elif 'raw' in trial:
                eeg_data = trial['raw'][()]
            elif 'data' in trial:
                eeg_data = trial['data'][()]
            else:
                # Find first dataset with 2D shape
                for k in trial.keys():
                    if isinstance(trial[k], h5py.Dataset) and len(trial[k].shape) >= 2:
                        eeg_data = trial[k][()]
                        break
        
        # Convert to tensor (Time, Channels)
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        return eeg_tensor
    
    def predict(self, eeg_data: torch.Tensor, max_length: int = 448) -> str:
        """
        Generate text from EEG data
        
        Args:
            eeg_data: EEG tensor of shape (Time, Channels)
            max_length: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        with torch.no_grad():
            # Add batch dimension and move to device
            eeg_batch = eeg_data.unsqueeze(0).to(self.device)  # (1, Time, Channels)
            
            # Encode EEG to Whisper embedding space
            eeg_embeddings = self.eeg_encoder(eeg_batch)  # (1, Time, Embed_dim)
            
            # Align to Whisper's expected sequence length (1500)
            target_len = 1500
            if eeg_embeddings.shape[1] != target_len:
                eeg_embeddings = torch.nn.functional.interpolate(
                    eeg_embeddings.transpose(1, 2),
                    size=target_len,
                    mode='linear'
                ).transpose(1, 2)
            
            # Get tokenizer
            tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=self.whisper_model.is_multilingual,
                language="en",
                task="transcribe"
            )
            
            # Simple greedy decoding
            # Start with SOT sequence
            sot_sequence = tokenizer.sot_sequence
            tokens = list(sot_sequence)
            
            for _ in range(max_length):
                # Convert tokens to tensor
                token_tensor = torch.tensor([tokens], device=self.device)
                
                # Run decoder
                logits = self.whisper_model.decoder(token_tensor, eeg_embeddings)
                
                # Get next token (greedy)
                next_token = logits[0, -1].argmax().item()
                
                # Check for end of text
                if next_token == tokenizer.eot:
                    break
                    
                tokens.append(next_token)
            
            # Decode tokens to text
            text = tokenizer.decode(tokens)
            
            return text
    
    def predict_from_file(self, hdf5_path: str, trial_name: str) -> str:
        """
        Generate text from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            trial_name: Name of trial
            
        Returns:
            Generated text
        """
        eeg_data = self.load_eeg_from_hdf5(hdf5_path, trial_name)
        return self.predict(eeg_data)
    
    def batch_predict(self, hdf5_files: List[str]) -> List[Dict]:
        """
        Generate text for multiple HDF5 files
        
        Args:
            hdf5_files: List of (hdf5_path, trial_name) tuples
            
        Returns:
            List of prediction results with metadata
        """
        results = []
        
        for hdf5_path, trial_name in hdf5_files:
            try:
                # Get ground truth if available
                ground_truth = None
                with h5py.File(hdf5_path, 'r') as f:
                    if trial_name in f and 'transcription' in f[trial_name]:
                        trans = f[trial_name]['transcription'][()]
                        if isinstance(trans, bytes):
                            ground_truth = trans.decode('utf-8')
                        elif isinstance(trans, np.ndarray):
                            ground_truth = ''.join([chr(c) for c in trans if c != 0])
                        else:
                            ground_truth = str(trans)
                        ground_truth = ground_truth.strip()
                
                # Generate prediction
                prediction = self.predict_from_file(hdf5_path, trial_name)
                
                results.append({
                    'file': hdf5_path,
                    'trial': trial_name,
                    'prediction': prediction,
                    'ground_truth': ground_truth
                })
                
            except Exception as e:
                print(f"Error processing {hdf5_path}/{trial_name}: {e}")
                results.append({
                    'file': hdf5_path,
                    'trial': trial_name,
                    'prediction': None,
                    'ground_truth': None,
                    'error': str(e)
                })
        
        return results
