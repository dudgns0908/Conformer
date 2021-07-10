from koasr.preprocess.ksponspeech import KsponSpeech
from koasr.preprocess.types import SpeechModeType, KsponSpeechVocabType

dataset_dict = {
    'ksponspeech': KsponSpeech
}


class Preprocess:
    def __init__(
            self,
            dataset_name: str,
            dataset_path: str,
            mode: str = 'phonetic',  # phonetic | spelling
    ):
        dataset_name = dataset_name.lower()
        assert dataset_name in dataset_dict.keys(), f'{dataset_name} dataset is not supported.'

        self.dataset_path = dataset_path
        self.preprocess_func = dataset_dict[dataset_name]
        self.mode = mode

    def _run_preprocess(self):
        audio_paths, transcripts = self.preprocess_func(dataset_path=self.dataset_path, mode=self.mode)
        return audio_paths, transcripts


def preprocess(
        dataset_name: str,
        script_file_dir: str,
        mode: str = SpeechModeType.PHOENTIC,
        save_manifest_path: str = './manifest.csv',
        vocab_path: str = './vocab.csv'
):
    dataset_name = dataset_name.lower()
    assert dataset_name in dataset_dict.keys(), f'{dataset_name} dataset is not supported.'
    # dataset_dict[dataset_name]
    preprocess_func = dataset_dict[dataset_name]().preprocess
    preprocess_func(
        script_file_dir=script_file_dir,
        mode=mode,
        vocab_type=KsponSpeechVocabType.GRAPHEME,
        manifest_file_path=save_manifest_path,
        vocab_path='./vocab.csv',
    )


preprocess('ksponspeech', '/Users/younghun/Data/ksponspeech/script')
