
import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Union
from transformers import WhisperTokenizer
from whisper_eeg import Config
from whisper_eeg.model import EEGWhisperModel

class EEGToTextInference:
    """Inference pipeline for EEG-to-Text model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config from checkpoint if available, otherwise try to find config.json or default
        saved_config = checkpoint.get('config', {})
        
        # Determine model type
        self.use_whisper = saved_config.get('use_whisper', True) # Default to True for now if missing
        
        print(f"Model type: {'Whisper' if self.use_whisper else 'Legcay'}")
        
        # Recreate model architecture
        # We need to know parameters. Ideally they should be in saved_config.
        # For now, we'll try to load from default.yaml if they are not full
        # But let's assume standard defaults if missing
        
        # Try to load full config from file to ensure we have latest architecture params
        # (Checkpoint might rely on old defaults if we don't be careful)
        try:
            self.config = Config.from_yaml('configs/default.yaml')
            print("Loaded default config for model parameters.")
        except Exception as e:
            print(f"Warning: Could not load default.yaml: {e}")
            self.config = None
        
        # Override defaults with saved_config if present, or fallback to self.config
        
        if self.use_whisper:
            whisper_model_name = saved_config.get('whisper_model_name', "openai/whisper-tiny")
            
            # Get dimensions from config if available
            eeg_dim = 512
            enc_layers = 6 # UPDATED DEFAULT
            heads = 8
            
            if self.config:
                eeg_dim = self.config.model.eeg_input_dim
                enc_layers = self.config.model.encoder_layers
                heads = self.config.model.num_heads
            
            # Initialize Tokenizer
            self.tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name, language="english", task="transcribe")
            
            self.model = EEGWhisperModel(
                whisper_model_name=whisper_model_name,
                eeg_input_dim=eeg_dim,
                encoder_layers=enc_layers, 
                num_heads=heads, 
            ).to(self.device)
        else:
             # Legacy
             print("Legacy model inference not fully supported in this updated script yet.")
             raise NotImplementedError
        
        # Load weights
        # Handle DataParallel keys (module.prefix)
        state_dict = checkpoint['model_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print("Model loaded successfully.")
    
    
    def generate_batch(self, eeg: torch.Tensor, max_length: int = None) -> List[str]:
        """Generate for a batch of EEG signals (Tensor)"""
        eeg = eeg.to(self.device)
        
        # Use config max length if not provided
        if max_length is None:
            if self.config:
                max_length = self.config.model.max_generation_length
            else:
                max_length = 200 # Fallback
        
        with torch.no_grad():
            if self.use_whisper:
                # Add repetition penalties to stop looping
                generated_ids = self.model.generate(
                    eeg, 
                    max_length=max_length,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                 # Legacy placeholder
                 texts = [""] * eeg.size(0)
        
        return texts

def run_inference(checkpoint: str, input_file: str = None, output_file: str = "predictions.csv", split: str = "test"):
    from tqdm import tqdm
    from whisper_eeg import create_dataloaders
    import csv
    
    # Initialize Inference Model
    try:
        inference = EEGToTextInference(checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint}")
        return

    # Determine Data Source
    loader = None
    
    if input_file:
        print(f"Processing single file: {input_file}")
        print("Single file loading not implemented yet. Please use default set.")
        return
    else:
        print(f"No input file provided. Loading standard {split.upper()} set from config...")
        # Load default config to get data path
        config = Config.from_yaml('configs/default.yaml')
        
        dataloaders = create_dataloaders(
            data_dir=config.data.data_dir,
            tokenizer=inference.tokenizer,
            batch_size=8, 
            num_workers=0,
            max_eeg_length=config.data.max_eeg_length,
            max_text_length=config.data.max_text_length
        )
        loader = dataloaders.get(split)
    
    if loader is None:
        print(f"No {split} data found.")
        return
        
    # Run Inference
    print(f"Generating predictions for {len(loader.dataset)} samples...")
    print(f"Saving to {output_file}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted"])
        
        for batch in tqdm(loader, desc="Inferencing"):
            eeg = batch['eeg']
            refs = batch['text_raw'] # Ground truth list
            
            # Generate
            preds = inference.generate_batch(eeg)
            
            # Save
            for ref, pred in zip(refs, preds):
                # Clean strings for CSV (remove newlines in text just in case)
                ref_clean = ref.replace('\n', ' ').strip()
                pred_clean = pred.replace('\n', ' ').strip()
                
                writer.writerow([ref_clean, pred_clean])
                
    print("Done!")
