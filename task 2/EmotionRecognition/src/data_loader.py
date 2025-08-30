import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDataLoader:
    """Data loader for emotion recognition datasets (RAVDESS, TESS, EMO-DB)"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sampling_rate = self.config['dataset']['sampling_rate']
        self.max_duration = self.config['dataset']['max_duration']
        self.emotion_map = self.config['emotions']
        
    def load_ravdess_dataset(self, data_dir: str) -> Tuple[List[str], List[int]]:
        """Load RAVDESS dataset"""
        audio_files = []
        emotions = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    # RAVDESS filename format: 03-01-06-01-02-01-12.wav
                    # Modality (01=full-AV, 02=video-only, 03=audio-only)
                    # Vocal channel (01=speech, 02=song)
                    # Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
                    # Emotional intensity (01=normal, 02=strong)
                    # Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
                    # Repetition (01=1st repetition, 02=2nd repetition)
                    # Actor (01 to 24)
                    
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = int(parts[2])
                        # Map RAVDESS emotion codes (1-8) to our emotion mapping (0-7)
                        emotion_idx = emotion_code - 1
                        if emotion_idx in self.emotion_map:
                            audio_files.append(os.path.join(root, file))
                            emotions.append(emotion_idx)
        
        return audio_files, emotions
    
    def load_tess_dataset(self, data_dir: str) -> Tuple[List[str], List[int]]:
        """Load TESS dataset"""
        audio_files = []
        emotions = []
        
        emotion_mapping = {
            'neutral': 0,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fear': 5,
            'disgust': 6,
            'ps': 7  # surprised
        }
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    filename = file.lower()
                    emotion_found = None
                    
                    for emotion_key, emotion_idx in emotion_mapping.items():
                        if emotion_key in filename:
                            emotion_found = emotion_idx
                            break
                    
                    if emotion_found is not None:
                        audio_files.append(os.path.join(root, file))
                        emotions.append(emotion_found)
        
        return audio_files, emotions
    
    def load_emodb_dataset(self, data_dir: str) -> Tuple[List[str], List[int]]:
        """Load EMO-DB dataset"""
        audio_files = []
        emotions = []
        
        emotion_mapping = {
            'W': 2,  # happy
            'L': 3,  # sad
            'A': 4,  # angry
            'F': 5,  # fearful
            'E': 6,  # disgust
            'T': 7   # surprised
        }
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    # EMO-DB filename format: [emotion][speaker][text][version].wav
                    emotion_code = file[0]
                    if emotion_code in emotion_mapping:
                        audio_files.append(os.path.join(root, file))
                        emotions.append(emotion_mapping[emotion_code])
        
        return audio_files, emotions
    
    def load_dataset(self, dataset_name: str, data_dir: str) -> Tuple[List[str], List[int]]:
        """Load specified dataset"""
        if dataset_name.lower() == 'ravdess':
            return self.load_ravdess_dataset(data_dir)
        elif dataset_name.lower() == 'tess':
            return self.load_tess_dataset(data_dir)
        elif dataset_name.lower() == 'emodb':
            return self.load_emodb_dataset(data_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def preprocess_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Preprocess audio file: load, resample, normalize, trim silence"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Remove silence
            if self.config['preprocessing']['remove_silence']:
                audio, _ = librosa.effects.trim(
                    audio, 
                    top_db=self.config['preprocessing']['silence_threshold'] * 100
                )
            
            # Normalize
            if self.config['preprocessing']['normalize']:
                audio = librosa.util.normalize(audio)
            
            # Pad or truncate to fixed length
            max_samples = int(self.max_duration * self.sampling_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            
            return audio
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def create_dataset(self, dataset_name: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create complete dataset with preprocessed audio and labels"""
        audio_files, emotions = self.load_dataset(dataset_name, data_dir)
        
        if not audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        X = []
        y = []
        
        logger.info(f"Processing {len(audio_files)} audio files...")
        
        for i, (audio_file, emotion) in enumerate(zip(audio_files, emotions)):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(audio_files)} files")
            
            audio = self.preprocess_audio(audio_file)
            if audio is not None:
                X.append(audio)
                y.append(emotion)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Dataset created with {len(X)} samples")
        return X, y

# Example usage
if __name__ == "__main__":
    loader = EmotionDataLoader()
    
    # Example: Load RAVDESS dataset
    try:
        X, y = loader.create_dataset('ravdess', 'data/raw/RAVDESS')
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique emotions: {np.unique(y)}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the dataset is downloaded and placed in data/raw/RAVDESS/")
