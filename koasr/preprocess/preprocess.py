from koasr.preprocess.ksponspeech import KsponSpeech
from koasr.preprocess.types import SpeechModeType, KsponSpeechVocabType

dataset_dict = {
    'ksponspeech': KsponSpeech
}


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
