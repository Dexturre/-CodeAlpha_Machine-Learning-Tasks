import numpy as np
import librosa
from typing import List, Tuple, Optional
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extraction for emotion recognition from speech"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature extractor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.sampling_rate = self.config['dataset']['sampling_rate']
    
    def extract_mfcc(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features from audio signal"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sampling_rate,
                n_mfcc=self.feature_config['n_mfcc'],
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length'],
                n_mels=self.feature_config['n_mels'],
                fmin=self.feature_config['fmin'],
                fmax=self.feature_config['fmax']
            )
            
            # Add delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Stack features
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # Normalize features if enabled
            if self.feature_config['normalize_features']:
                features = self.normalize_features(features)
            
            return features.T  # Transpose to get (time_steps, features)
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract Mel spectrogram features"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length'],
                n_mels=self.feature_config['n_mels'],
                fmin=self.feature_config['fmin'],
                fmax=self.feature_config['fmax']
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            if self.feature_config['normalize_features']:
                log_mel_spec = self.normalize_features(log_mel_spec)
            
            return log_mel_spec.T
            
        except Exception as e:
            logger.error(f"Error extracting Mel spectrogram: {e}")
            return None
    
    def extract_spectral_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract various spectral features"""
        try:
            features = []
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            features.append(spectral_centroid)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            features.append(spectral_bandwidth)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            features.append(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            features.append(zcr)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            features.append(chroma)
            
            # Stack all features
            all_features = np.vstack(features)
            
            if self.feature_config['normalize_features']:
                all_features = self.normalize_features(all_features)
            
            return all_features.T
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return None
    
    def extract_features(self, audio: np.ndarray, feature_type: Optional[str] = None) -> Optional[np.ndarray]:
        """Extract features based on specified type"""
        if feature_type is None:
            feature_type = self.feature_config['feature_type']
        
        # Ensure feature_type is not None before calling .lower()
        if feature_type is None:
            raise ValueError("Feature type cannot be None")
        
        feature_type_lower = feature_type.lower()
        
        if feature_type_lower == 'mfcc':
            return self.extract_mfcc(audio)
        elif feature_type_lower == 'melspectrogram':
            return self.extract_mel_spectrogram(audio)
        elif feature_type_lower == 'spectral':
            return self.extract_spectral_features(audio)
        elif feature_type_lower == 'chroma':
            return self.extract_chroma_features(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def extract_chroma_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sampling_rate,
                n_fft=self.feature_config['n_fft'],
                hop_length=self.feature_config['hop_length']
            )
            
            if self.feature_config['normalize_features']:
                chroma = self.normalize_features(chroma)
            
            return chroma.T
            
        except Exception as e:
            logger.error(f"Error extracting chroma features: {e}")
            return None
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using mean and standard deviation"""
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std = np.where(std == 0, 1e-10, std)  # Avoid division by zero
        return (features - mean) / std
    
    def extract_features_batch(self, audio_batch: List[np.ndarray], feature_type: Optional[str] = None) -> List[np.ndarray]:
        """Extract features for a batch of audio signals"""
        features = []
        for i, audio in enumerate(audio_batch):
            if i % 100 == 0:
                logger.info(f"Extracting features for sample {i}/{len(audio_batch)}")
            
            feature = self.extract_features(audio, feature_type)
            if feature is not None:
                features.append(feature)
        
        return features
    
    def get_feature_shape(self) -> Tuple[int, int]:
        """Get the expected feature shape based on configuration"""
        if self.feature_config['feature_type'].lower() == 'mfcc':
            # MFCC + delta + delta-delta
            n_features = self.feature_config['n_mfcc'] * 3
        elif self.feature_config['feature_type'].lower() == 'melspectrogram':
            n_features = self.feature_config['n_mels']
        elif self.feature_config['feature_type'].lower() == 'spectral':
            # centroid + bandwidth + rolloff + zcr + chroma (12)
            n_features = 1 + 1 + 1 + 1 + 12
        elif self.feature_config['feature_type'].lower() == 'chroma':
            n_features = 12
        else:
            n_features = self.feature_config['n_mfcc']
        
        # Calculate time steps based on audio duration and hop length
        max_samples = int(self.config['dataset']['max_duration'] * self.sampling_rate)
        n_frames = max_samples // self.feature_config['hop_length'] + 1
        
        return n_frames, n_features

# Example usage
if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Create a test audio signal
    test_audio = np.random.randn(16000)  # 1 second of random audio
    
    # Extract different types of features
    mfcc_features = extractor.extract_mfcc(test_audio)
    mel_features = extractor.extract_mel_spectrogram(test_audio)
    spectral_features = extractor.extract_spectral_features(test_audio)
    
    # Handle Optional return types
    if mfcc_features is not None:
        print(f"MFCC features shape: {mfcc_features.shape}")
    else:
        print("MFCC features extraction failed")
    
    if mel_features is not None:
        print(f"Mel spectrogram shape: {mel_features.shape}")
    else:
        print("Mel spectrogram extraction failed")
    
    if spectral_features is not None:
        print(f"Spectral features shape: {spectral_features.shape}")
    else:
        print("Spectral features extraction failed")
    
    print(f"Expected feature shape: {extractor.get_feature_shape()}")
