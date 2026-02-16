"""Audio generation from text using TTS"""

import os
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from TTS.api import TTS
import torch
import pdb

class AudioGenerator:
    """Generate synthetic audio from text transcriptions using TTS"""
    
    def __init__(self, tts_model="tts_models/en/ljspeech/tacotron2-DDC", use_gpu=True):
        """
        Initialize TTS model
        
        Args:
            tts_model: TTS model name
            use_gpu: Whether to use GPU if available
        """
        
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
                                file_path=output_file,
                                language="en"
                            )
                            count += 1
                        except Exception as e:
                            print(f"Error generating audio for {trial_name}: {e}")
                            
            except Exception as e:
                print(f"Error processing {hdf5_file}: {e}")
        
def convert_audio_to_16khz(data_dir="data"):
    """Convert all audio files in directory to 16kHz"""
    import soundfile as sf
    import librosa
    from pathlib import Path
    
    print(f"Searching for audio files in {data_dir}...")
    audio_files = sorted(Path(data_dir).rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files.")
    
    count = 0
    for audio_path in tqdm(audio_files, desc="Converting audio"):
        try:
            # Read audio
            # soundfile matches librosa's behavior but is faster for read/write
            # however librosa.resample needs numpy array
            audio, sr = sf.read(str(audio_path))
            
            if sr != 16000:
                # Resample
                # librosa expects (channels, time) but soundfile returns (time, channels)
                # librosa.resample handles 1D or 2D. 
                # If stereo, it resamples each channel.
                
                # Check if stereo (Time, Channels)
                is_stereo = len(audio.shape) > 1
                if is_stereo:
                    audio = audio.T # Convert to (Channels, Time) for librosa
                    
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                if is_stereo:
                    audio_16k = audio_16k.T # Convert back to (Time, Channels) for soundfile
                
                # Write back to same path (overwrite)
                sf.write(str(audio_path), audio_16k, 16000)
                count += 1
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            
    print(f"Converted {count} files to 16kHz.")

if __name__ == "__main__":
    # If run directly, run conversion
    convert_audio_to_16khz()