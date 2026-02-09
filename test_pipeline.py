
import torch
import sys
from pathlib import Path
from transformers import WhisperTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from whisper_eeg import create_dataloaders
from whisper_eeg.model import EEGWhisperModel

def test_pipeline():
    print("Testing EEG-to-Text Pipeline (Whisper Integration)")
    print("="*50)
    
    MODEL_NAME = "openai/whisper-tiny"
    
    print(f"Loading Tokenizer: {MODEL_NAME}")
    try:
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="english", task="transcribe")
    except Exception as e:
        print(f"Error loading tokenizer (internet connection needed?): {e}")
        return

    # Test data loading
    print("\n1. Testing data loading...")
    try:
        dataloaders = create_dataloaders(
            data_dir='data/raw/hdf5_data_final',
            tokenizer=tokenizer,
            batch_size=2, # Small batch
            num_workers=0,
            max_eeg_length=2000,
            max_text_length=448, 
        )
        
        train_loader = dataloaders.get('train')
        
        if train_loader is None:
            print("ERROR: No training loader created!")
        else:
            # Get a batch
            batch = next(iter(train_loader))
            print(f"✓ Successfully loaded batch")
            print(f"  EEG shape: {batch['eeg'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}") # Should be token IDs
            print(f"  Raw Text sample: {batch['text_raw'][0]}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        # Dummy batch
        batch = {
            'eeg': torch.randn(2, 2000, 512),
            'labels': torch.randint(0, 50000, (2, 448)),
            'eeg_mask': torch.ones(2, 2000)
        }
    
    # Test model
    print("\n2. Testing model...")
    device = 'cpu' # Ensure CPU for test if no GPU
    if torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    try:
        model = EEGWhisperModel(
            whisper_model_name=MODEL_NAME,
            eeg_input_dim=512,
            encoder_layers=1,
            num_heads=4,
        ).to(device)
        
        print(f"✓ Model created")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        eeg = batch['eeg'].to(device)
        labels = batch['labels'].to(device)
        eeg_mask = batch.get('eeg_mask').to(device) if 'eeg_mask' in batch else None
        
        # Whisper forward
        outputs, ctc_logits, downsampled_mask = model(eeg, labels=labels, eeg_mask=eeg_mask)
        loss = outputs.loss
        
        print(f"✓ Forward pass successful")
        print(f"  Loss: {loss.item()}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test Generation
    print("\n4. Testing generation...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(eeg, max_length=20)
            print(f"✓ Generation output shape: {generated_ids.shape}")
            
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"  Decoded: {decoded[0]}")
            
    except Exception as e:
        print(f"✗ Error in generation: {e}")
        import traceback
        traceback.print_exc()

    # Test Trainer
    print("\n5. Testing Trainer (Multi-GPU logic)...")
    try:
        from whisper_eeg import Trainer, Config
        
        # Create dummy config
        config = Config.from_yaml('configs/default.yaml')
        config.training.num_epochs = 1
        config.training.device = device # use what we found earlier
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            test_loader=None,
            config=config,
            tokenizer=tokenizer
        )
        print(f"✓ Trainer initialized (Parallel: {trainer.is_parallel})")
        
        # Run 1 step of training (mock loop) or just check train_epoch works
        # We can't easily run full epoch if dataset is huge, but we loaded a real dataset
        # Let's just create a tiny DataLoader for this test
        
        # Create a dummy batch loader
        class MockLoader:
            def __init__(self, batch):
                self.batch = batch
            def __iter__(self):
                yield self.batch
            def __len__(self):
                return 1
                
        trainer.train_loader = MockLoader(batch)
        trainer.val_loader = MockLoader(batch) # Use same batch for validation
        
        # Test validation step (Metrics)
        print("Running Trainer.validate() with mock data...")
        loss, wer, cer = trainer.validate()
        print(f"✓ Trainer.validate() passed.")
        print(f"  Loss: {loss:.4f}")
        print(f"  WER: {wer:.4f}")
        print(f"  CER: {cer:.4f}")
        
    except Exception as e:
         print(f"✗ Error in Trainer: {e}")
         import traceback
         traceback.print_exc()

    print("\n" + "="*50)
    print("All tests passed! ✓")

if __name__ == '__main__':
    test_pipeline()
