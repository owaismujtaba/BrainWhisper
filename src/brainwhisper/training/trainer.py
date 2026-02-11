"""Training logic for EEG encoder with multi-GPU support"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..models import EEGEncoder
from ..dataset import EEGDataset, collate_fn


class Trainer:
    """Trainer for EEG Encoder using Teacher-Student learning with multi-GPU support"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup device
        if torch.cuda.is_available() and config.training.device == "cuda":
            self.device = torch.device("cuda")
            self.use_multi_gpu = config.training.distributed and torch.cuda.device_count() > 1
            if self.use_multi_gpu:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            else:
                print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.use_multi_gpu = False
            print("Using CPU")
        
        # Load Teacher (Whisper)
        print("Loading Whisper Teacher model...")
        self.whisper_model = whisper.load_model(config.model.whisper_model, device=self.device)
        self.teacher_encoder = self.whisper_model.encoder
        self.teacher_encoder.eval()
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        
        # Wrap teacher with DataParallel if using multi-GPU
        if self.use_multi_gpu:
            self.teacher_encoder = nn.DataParallel(self.teacher_encoder)
        
        # Initialize Student (EEG Encoder)
        print("Initializing EEG Encoder Student...")
        embed_dim = self.whisper_model.dims.n_audio_state
        self.student = EEGEncoder(
            input_channels=config.model.eeg_channels,
            output_dim=embed_dim,
            hidden_dim=config.model.hidden_dim,
            kernel_size=config.model.kernel_size,
            stride=config.model.stride,
            num_layers=config.model.num_cnn_layers,
            dropout=config.model.dropout
        ).to(self.device)
        
        # Wrap student with DataParallel if using multi-GPU
        if self.use_multi_gpu:
            self.student = nn.DataParallel(self.student)
            self.model_without_parallel = self.student.module
        else:
            self.model_without_parallel = self.student
        
        # Setup data
        self.train_dataset = EEGDataset(
            config.paths.data_dir,
            config.paths.audio_dir,
            split=config.data.train_split,
            limit=config.data.limit
        )
        self.val_dataset = EEGDataset(
            config.paths.data_dir,
            config.paths.audio_dir,
            split=config.data.val_split,
            limit=config.data.limit
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=max(1, config.training.batch_size // 2),  # Use half batch size for validation
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing for validation
            pin_memory=False  # Disable pin_memory to avoid potential issues
        )
        
        # Optimization
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.student.parameters(), lr=config.training.learning_rate)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}")
        
        for eeg, mel in pbar:
            eeg = eeg.to(self.device, non_blocking=True)
            mel = mel.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Get Teacher Targets
            with torch.no_grad():
                target_emb = self.teacher_encoder(mel)
            
            # Student Prediction
            pred_emb = self.student(eeg)
            
            # Align Dimensions (Interpolate Student to Match Teacher)
            if pred_emb.shape[1] != target_emb.shape[1]:
                pred_emb = torch.nn.functional.interpolate(
                    pred_emb.transpose(1, 2),
                    size=target_emb.shape[1],
                    mode='linear'
                ).transpose(1, 2)
            
            loss = self.criterion(pred_emb, target_emb)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for eeg, mel in tqdm(self.val_loader, desc="Validating", leave=False):
                eeg = eeg.to(self.device, non_blocking=True)
                mel = mel.to(self.device, non_blocking=True)
                
                # Teacher forward
                teacher_features = self.teacher_encoder(mel)
                
                # Student forward
                student_features = self.student(eeg)
                
                # Align temporal dimension if needed
                if student_features.shape[1] != teacher_features.shape[1]:
                    student_features = torch.nn.functional.interpolate(
                        student_features.transpose(1, 2),
                        size=teacher_features.shape[1],
                        mode='linear'
                    ).transpose(1, 2)
                
                loss = self.criterion(student_features, teacher_features)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model_without_parallel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        path = os.path.join(self.config.paths.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if self.config.training.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_eeg_encoder.pth")
                print("Saved best model.")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Always save last
            self.save_checkpoint("last_eeg_encoder.pth")
        
        print("Training complete!")
