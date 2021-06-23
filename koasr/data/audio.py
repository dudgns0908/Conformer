import os

import librosa
import numpy as np


def load_audio(path: str, sampling_rate: int = 16000, del_silence: bool = False):
    try:
        extension = os.path.splitext(path)[-1][1:]
        if extension in ('wav',):
            signal, _ = librosa.load(path, sr=sampling_rate)
        elif extension == 'pcm':
            signal = np.memmap(path, dtype='h', mode='r').astype('float32')

            if del_silence:
                non_silence_indices = librosa.effects.split(signal, top_db=30)
                signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

            signal /= 32767  # normalize audio
        else:
            raise ValueError(f'This extension ({extension}) is not supported')

        return signal

    except FileNotFoundError as e:
        raise e
    except ValueError as e:
        raise e


if __name__ == '__main__':
    load_audio('../../datasets/KsponSpeech/valid/01')