"""Audio generation from text using TTS"""

import os
import h5py
from pathlib import Path
from tqdm import tqdm


class AudioGenerator:
    """Generate synthetic audio from text transcriptions using TTS"""
    
    def __init__(self, tts_model="tts_models/en/ljspeech/tacotron2-DDC", use_gpu=True):
        """
        Initialize TTS model
        
        Args:
            tts_model: TTS model name
            use_gpu: Whether to use GPU if available
        """
        from TTS.api import TTS
        import torch
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        print("Initializing TTS model...")
        try:
            self.tts = TTS(model_name=tts_model, progress_bar=False, gpu=self.use_gpu)
        except Exception as e:
            print(f"Error initializing TTS model: {e}")
            raise
    
    def generate_from_hdf5(self, data_dir, output_dir, split="train", limit=None):
        """
        Generate audio files from HDF5 transcriptions
        
        Args:
            data_dir: Directory containing HDF5 files
            output_dir: Output directory for audio files
            split: Data split ('train' or 'val')
            limit: Optional limit on number of files to process
        """
        # Find HDF5 files
        search_path = os.path.join(data_dir, split) if os.path.exists(os.path.join(data_dir, split)) else data_dir
        pattern = f"data_{split}.hdf5"
        hdf5_files = sorted(Path(search_path).rglob(pattern))
        
        if not hdf5_files:
            print(f"Warning: Split folder '{split}' not found. Searching in {search_path}...")
            hdf5_files = sorted(Path(search_path).rglob("*.hdf5"))
        
        print(f"Found {len(hdf5_files)} HDF5 files.")
        
        count = 0
        
        for hdf5_file in hdf5_files:
            if limit and count >= limit:
                print(f"Reached limit of {limit} files.")
                break
            
            # Get subject name from parent directory
            subject_name = hdf5_file.parent.name
            
            # Create subject-specific output directory
            subject_output_dir = os.path.join(output_dir, split, subject_name)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            # Read HDF5 file
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    trials = list(f.keys())
                    
                    for trial_name in tqdm(trials, desc=f"Processing {subject_name}/{hdf5_file.name}"):
                        trial = f[trial_name]
                        
                        # Get transcription
                        if 'transcription' not in trial:
                            continue
                        
                        transcription = trial['transcription'][()]
                        
                        # Decode if bytes
                        if isinstance(transcription, bytes):
                            text = transcription.decode('utf-8')
                        elif isinstance(transcription, np.ndarray):
                            # Convert from ASCII codes to string
                            text = ''.join([chr(c) for c in transcription if c != 0])
                        else:
                            text = str(transcription)
                        
                        text = text.strip()
                        
                        if not text:
                            continue
                        
                        # Create safe filename
                        safe_name = "".join(c for c in trial_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(" ", "_")
                        output_file = os.path.join(subject_output_dir, f"{safe_name}.wav")
                        
                        # Skip if already exists
                        if os.path.exists(output_file):
                            continue
                        
                        # Generate audio using TTS
                        try:
                            self.tts.tts_to_file(
                                text=text,
                                file_path=output_file
                            )
                            count += 1
                        except Exception as e:
                            print(f"Error generating audio for {trial_name}: {e}")
                            
            except Exception as e:
                print(f"Error processing {hdf5_file}: {e}")
        
        print(f"Finished! Generated {count} audio files in {os.path.join(output_dir, split)}")
