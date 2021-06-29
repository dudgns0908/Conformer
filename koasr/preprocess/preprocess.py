from koasr.preprocess.ksponspeech import preprocess_ksponspeech

dataset_dict = {
    'ksponspeech': preprocess_ksponspeech
}


def preprocess(
        dataset_name: str,
        dataset_path: str,
        mode: str = 'phonetic'
):
    dataset_name = dataset_name.lower()
    assert dataset_name in dataset_dict.keys(), f'{dataset_name} dataset is not supported.'

    import time

    preprocess_func = dataset_dict[dataset_name]

    start = time.time()
    return print(preprocess_func(dataset_path=dataset_path, mode=mode)), print(time.time() - start)


preprocess('ksponspeech', '/Users/younghun/Data/ksponspeech/train/')
