"""Training module for EEG encoder"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import wandb

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
        
        # Setup logging
        log_dir = Path(config.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "train.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Starting training session - Log file: {log_file}")

        # Setup device
        if torch.cuda.is_available() and config.training.device == "cuda":
            self.device = torch.device("cuda")
            self.use_multi_gpu = config.training.distributed and torch.cuda.device_count() > 1
            if self.use_multi_gpu:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            else:
                self.logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.use_multi_gpu = False
            self.logger.info("Using CPU")
        
        # Initialize Weights & Biases
        self.use_wandb = hasattr(config, 'wandb') and config.wandb.enabled
        if self.use_wandb:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                config={
                    "model": {
                        "eeg_channels": config.model.eeg_channels,
                        "whisper_model": config.model.whisper_model,
                        "hidden_dim": config.model.hidden_dim,
                        "num_cnn_layers": config.model.num_cnn_layers,
                        "kernel_size": config.model.kernel_size,
                        "stride": config.model.stride,
                        "dropout": config.model.dropout,
                        "num_transformer_layers": config.model.num_transformer_layers,
                        "num_attention_heads": config.model.num_attention_heads,
                    },
                    "training": {
                        "batch_size": config.training.batch_size,
                        "epochs": config.training.epochs,
                        "learning_rate": config.training.learning_rate,
                        "num_workers": config.training.num_workers,
                        "distributed": config.training.distributed,
                    },
                    "data": {
                        "train_split": config.data.train_split,
                        "val_split": config.data.val_split,
                        "train_samples": 0,  # Will update after dataset loading
                        "val_samples": 0,
                    }
                }
            )
            self.logger.info(f"Weights & Biases initialized - Project: {config.wandb.project}")
        else:
            self.logger.info("Weights & Biases disabled")
        
        # Load Teacher (Whisper)
        self.logger.info("Loading Whisper Teacher model...")
        self.whisper_model = whisper.load_model(config.model.whisper_model, device=self.device)
        self.teacher_encoder = self.whisper_model.encoder
        self.teacher_encoder.eval()
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        
        # Wrap teacher with DataParallel if using multi-GPU
        if self.use_multi_gpu:
            self.teacher_encoder = nn.DataParallel(self.teacher_encoder)
        
        # Initialize Student (EEG Encoder)
        self.logger.info("Initializing EEG Encoder Student...")
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
        self.logger.info(f"Scanning dataset for split '{config.data.train_split}'...")
        self.train_dataset = EEGDataset(
            config.paths.data_dir,
            config.paths.audio_dir,
            split=config.data.train_split,
            limit=config.data.limit
        )
        self.logger.info(f"Found {len(self.train_dataset)} samples.")
        
        self.logger.info(f"Scanning dataset for split '{config.data.val_split}'...")
        self.val_dataset = EEGDataset(
            config.paths.data_dir,
            config.paths.audio_dir,
            split=config.data.val_split,
            limit=config.data.limit
        )
        self.logger.info(f"Found {len(self.val_dataset)} samples.")
        
        # Update wandb config with dataset sizes
        if self.use_wandb:
            wandb.config.update({
                "data/train_samples": len(self.train_dataset),
                "data/val_samples": len(self.val_dataset)
            })
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        # Calculate validation batch size (must be divisible by number of GPUs)
        num_gpus = torch.cuda.device_count() if self.use_multi_gpu else 1
        val_batch_size = max(num_gpus, (config.training.batch_size // 2) // num_gpus * num_gpus)
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
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
        # Use unwrapped model for validation to avoid DataParallel issues
        model_to_validate = self.model_without_parallel
        model_to_validate.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for eeg, mel in tqdm(self.val_loader, desc="Validating", leave=False):
                eeg = eeg.to(self.device, non_blocking=True)
                mel = mel.to(self.device, non_blocking=True)
                
                # Teacher forward
                teacher_features = self.teacher_encoder(mel)
                
                # Student forward (using unwrapped model)
                student_features = model_to_validate(eeg)
                
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
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model_without_parallel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs: {self.config.training.epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}/{self.config.training.epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/best_loss": self.best_val_loss
                }, step=epoch)
            
            # Save best model
            if self.config.training.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config.paths.checkpoint_dir, "best_eeg_encoder.pth")
                self.save_checkpoint(best_path)
                self.logger.info(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
                
                # Log best model to wandb
                if self.use_wandb:
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_epoch"] = epoch
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pth"
                )
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(self.config.paths.checkpoint_dir, "last_eeg_encoder.pth")
        self.save_checkpoint(final_path)
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 60)
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

