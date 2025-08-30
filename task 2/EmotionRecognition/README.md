# Emotion Recognition from Speech

A deep learning system that recognizes human emotions from speech audio using MFCC features and neural networks.

## Features
- MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
- CNN, RNN, and LSTM model architectures
- Support for multiple datasets: RAVDESS, TESS, EMO-DB
- Data preprocessing and augmentation
- Model training and evaluation
- Real-time emotion prediction

## Project Structure
```
EmotionRecognition/
├── data/                    # Dataset storage
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── feature_extraction.py # MFCC feature extraction
│   ├── models.py           # Neural network models
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── utils.py           # Utility functions
├── models/                 # Trained model weights
├── requirements.txt       # Python dependencies
└── config.yaml            # Configuration file
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Download datasets (RAVDESS, TESS, or EMO-DB)
2. Preprocess data: `python src/data_loader.py`
3. Train model: `python src/train.py`
4. Predict emotions: `python src/predict.py --audio path/to/audio.wav`

## Supported Emotions
- Happy
- Angry
- Sad
- Neutral
- Fear
- Disgust
- Surprise
