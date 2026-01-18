import librosa
import numpy as np
import pandas as pd

class AudioProcessor:
    @staticmethod
    def extract_features(y, sr):
        features = {
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_contrast': float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
            'mfcc': float(np.mean(librosa.feature.mfcc(y=y, sr=sr))),
            'chroma': float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        }
        return pd.DataFrame(features, index=[0])
