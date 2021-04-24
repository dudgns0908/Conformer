import os

import librosa
import numpy as np


def load_audio(path: str, sampling_rate: int = 16000):
    try:
        extension = os.path.splitext(path)[-1][1:]
        if extension in ('wav',):
            signal, _ = librosa.load(path, sr=sampling_rate)
        elif extension == 'pcm':
            signal = np.memmap(path, dtype='h', mode='r').astype('float32')
        else:
            raise ValueError(f'This extension ({extension}) is not supported')

    except FileNotFoundError as e:
        raise e


if __name__ == '__main__':
    load_audio('./test.wav')
