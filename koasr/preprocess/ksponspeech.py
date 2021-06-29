import os
import re

import ray

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm
import time

PERCENT_FILES = {
    '087797': '퍼센트',
    '215401': '퍼센트',
    '284574': '퍼센트',
    '397184': '퍼센트',
    '501006': '프로',
    '502173': '프로',
    '542363': '프로',
    '581483': '퍼센트'
}


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def get_filtered_sentence(file_path, mode):
    with open(file_path, 'r', encoding='cp949') as f:
        sentence = f.read()
        file_name = os.path.basename(file_path)
        replace = PERCENT_FILES.get(file_name[12:18], None)
        return special_filter(bracket_filter(sentence, mode), mode, replace)


def preprocess_ksponspeech(dataset_path, mode='phonetic'):
    print('preprocess started..')

    audio_paths = list()
    transcripts = list()
    with Parallel(n_jobs=cpu_count() - 1) as parallel:
        for folder in os.listdir(dataset_path):  # {KsponSpeech_01, ..., KsponSpeech_05}
            path = os.path.join(dataset_path, folder)
            if not folder.startswith('KsponSpeech') or not os.path.isdir(path):
                continue

            for idx, subfolder in tqdm(enumerate(os.listdir(path)), desc=f'Preprocess text files on {path}'):
                path = os.path.join(dataset_path, folder, subfolder)
                if not os.path.isdir(path):
                    continue

                text_file_list = []
                audio_sub_file_list = []
                relative_path = os.path.join(folder, subfolder)
                for file_name in os.listdir(path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(path, file_name)
                        text_file_list.append(file_path)
                        audio_sub_file_list.append(os.path.join(relative_path, file_name))

                new_sentence = parallel(delayed(get_filtered_sentence)(s, mode) for s in text_file_list)
                transcripts.extend(new_sentence)
                audio_paths.extend(audio_sub_file_list)

    return audio_paths, transcripts
