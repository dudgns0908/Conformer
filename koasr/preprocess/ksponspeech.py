import os
import ray

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm


def preprocess_ksponspeech(dataset_path, mode='phonetic'):
    print('preprocess started..')

    audio_paths = list()
    transcripts = list()

    with Parallel(n_jobs=cpu_count() - 1) as parallel:
        for folder in os.listdir(dataset_path):
            # folder : {KsponSpeech_01, ..., KsponSpeech_05}
            path = os.path.join(dataset_path, folder)
            if not folder.startswith('KsponSpeech') or not os.path.isdir(path):
                continue

            subfolders = os.listdir(path)
            for idx, subfolder in tqdm(list(enumerate(subfolders)), desc=f'Preprocess text files on {path}'):
                path = os.path.join(dataset_path, folder, subfolder)
                if not os.path.isdir(path):
                    continue

                # list-up files
                sub_file_list = []
                audio_sub_file_list = []
                for file_name in os.listdir(path):
                    if file_name.endswith('.txt'):
                        sub_file_list.append(os.path.join(path, file_name))
                        audio_sub_file_list.append(os.path.join(folder, subfolder, file_name))

                # do parallel and get results
                # new_sentences = parallel(
                #     delayed(read_preprocess_text_file)(p, mode) for p in sub_file_list
                # )

                audio_paths.extend(audio_sub_file_list)
                transcripts.extend(new_sentences)

    return audio_paths, transcripts