from koasr.preprocess.ksponspeech import KsponSpeech

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
        dataset_path: str,
        mode: str = 'phonetic',
        save_path: str = './'
):
    KsponSpeech().preprocess(
        dataset_path,
        script_file_dir=f'/Users/younghun/Data/ksponspeech/script',

    )
    # dataset_name = dataset_name.lower()
    # assert dataset_name in dataset_dict.keys(), f'{dataset_name} dataset is not supported.'
    #
    # preprocess_func = dataset_dict[dataset_name]
    # audio_paths, transcripts = preprocess_func(dataset_path=dataset_path, mode=mode)

    return None


preprocess('ksponspeech', '/Users/younghun/Data/ksponspeech/train/')
