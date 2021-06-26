import os
from typing import Tuple, Any, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from koasr.data.audio import load_audio


class AudioDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            audio_paths: list,
            transcripts: list,
            sampling_rate: int = 16000,
            sos_id: int = 1,
            eos_id: int = 2,
            del_silence: bool = False,
            spec_augment: bool = False,
    ) -> None:
        super().__init__()
        assert len(audio_paths) == len(transcripts), 'audio_paths and transcripts must be the same length.'

        self.dataset_dir = dataset_dir
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.sampling_rate = sampling_rate
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.del_silence = del_silence
        self.spec_augment = spec_augment

    def __getitem__(self, index) -> Tuple[Tensor, Union[list, None]]:
        data_path = os.path.join(self.dataset_dir, self.audio_paths[index])
        data = self._get_audio_feature(data_path, None)
        transcript = None if self.transcripts is None else self.transcripts[index]

        return data, transcript

    def _get_audio_feature(self, path: str, augment):
        signal = load_audio(path, del_silence=self.del_silence)
        return signal
        # feature = self.transforms(signal)
        # feature -= feature.mean()
        # feature /= np.std(feature)
        # feature = torch.FloatTensor(feature).transpose(0, 1)

        # if augment == self.SPEC_AUGMENT:
        #     feature = self._spec_augment(feature)

        # return feature


if __name__ == '__main__':

    dataset = AudioDataset(
        dataset_path='/Users/younghun/Data/ksponspeech/train',
        audio_paths=['KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000104.pcm'],
        transcripts=['아 몬 소리야, 그건 또']
    )

    for d, f in dataset:
        print(d, f)