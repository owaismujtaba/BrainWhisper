
import argparse
import sys
from transformers import WhisperTokenizer
from whisper_eeg import Config, Trainer, create_dataloaders
from whisper_eeg.model import EEGWhisperModel # Updated import

def train(config_path: str):
    """Run training with valid config path"""
    config = Config.from_yaml(config_path)
    
    print("Configuration loaded:")
    print(config)
    
    tokenizer = None
    if config.model.use_whisper:
        print(f"Loading Whisper Tokenizer: {config.model.whisper_model_name}")
        tokenizer = WhisperTokenizer.from_pretrained(config.model.whisper_model_name, language="english", task="transcribe")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_dir=config.data.data_dir,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        max_eeg_length=config.data.max_eeg_length,
        max_text_length=config.data.max_text_length,
        gaussian_noise=config.data.gaussian_noise,
    )
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')
    
    if train_loader is None:
        raise ValueError("No training data found!")
    
    # Create model
    print("\nCreating model...")
    if config.model.use_whisper:
        model = EEGWhisperModel(
            whisper_model_name=config.model.whisper_model_name,
            eeg_input_dim=config.model.eeg_input_dim,
            encoder_layers=config.model.encoder_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            freeze_decoder=config.model.freeze_decoder,
        )
    else:
        raise ValueError("Legacy EEGToTextModel has been removed. Please set use_whisper=True in config.")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        tokenizer=tokenizer, 
    )
    
    # Train
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Train EEG-to-Text Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)

if __name__ == "__main__":
    main()
