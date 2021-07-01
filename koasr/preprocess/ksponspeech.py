import os
import re
import unicodedata
import pandas as pd
from typing import Union

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


def sentence_filter(sentence, mode, replace):
    return special_filter(bracket_filter(sentence, mode), mode, replace)


class KsponSpeech:
    train_trn = ('train.trn', )
    eval_trn = ("eval_clean.trn", "eval_other.trn")

    def preprocess(
            self,
            dataset_path: str,
            script_file_dir: str,
            mode: str = 'phonetic',
    ):

        train_audio_paths, train_transcripts = self.preprocess_sentence(script_file_dir, self.train_trn, mode)
        eval_audio_paths, eval_transcripts = self.preprocess_sentence(script_file_dir, self.eval_trn, mode)

        audio_paths = train_audio_paths + eval_audio_paths
        transcripts = train_transcripts + eval_transcripts

        manifest_file_path: str = './'
        vocab_path: str = './'
        self.sentence_to_grapheme(audio_paths, transcripts ,manifest_file_path, vocab_path)
        # if self.configs.vocab.unit == 'kspon_character':
        #     generate_character_labels(transcripts, self.configs.vocab.vocab_path)
        #     generate_character_script(audio_paths, transcripts, manifest_file_path, self.configs.vocab.vocab_path)
        #
        # elif self.configs.vocab.unit == 'kspon_subword':
        #     train_sentencepiece(transcripts, self.configs.vocab.vocab_size, self.configs.vocab.blank_token)
        #     sentence_to_subwords(
        #         audio_paths, transcripts, manifest_file_path, sp_model_path=self.configs.vocab.sp_model_path
        #     )
        #
        # elif self.configs.vocab.unit == 'kspon_grapheme':
        #     sentence_to_grapheme(audio_paths, transcripts, manifest_file_path, self.configs.vocab.vocab_path)
        #
        # else:
        #     raise ValueError(f"Unsupported vocab : {self.configs.vocab.unit}")

    def preprocess_sentence(
            self,
            script_file_dir: str,
            script_file_name: Union[str, tuple, list],
            mode: str = 'phonetic'
    ):
        script_names = [script_file_name] if isinstance(script_file_name, str) else script_file_name

        audio_paths = []
        transcripts = []
        for script_name in script_names:
            print(f'star preprocess {script_name}')
            with open(os.path.join(script_file_dir, script_name), 'r') as f:
                for line in tqdm(f.readlines()):
                    audio_path, raw_transcript = line.split(" :: ")
                    audio_paths.append(audio_path)

                    file_name = os.path.basename(audio_path)
                    replace = PERCENT_FILES.get(file_name[12:18], None)
                    transcript = sentence_filter(raw_transcript, mode=mode, replace=replace)
                    transcripts.append(transcript)

        return audio_paths, transcripts

    def sentence2unit(self, unit='grapheme'):
        pass

    def sentence_to_grapheme(self, audio_paths, transcripts, manifest_file_path: str, vocab_path: str):
        grapheme_transcripts = list()

        for transcript in transcripts:
            grapheme_transcripts.append(" ".join(unicodedata.normalize('NFKD', transcript).replace(' ', '|')).upper())

        generate_grapheme_labels(grapheme_transcripts, vocab_path)

        print('create_script started..')
        grpm2id, id2grpm = load_label(vocab_path)

        with open(manifest_file_path, "w") as f:
            for audio_path, transcript, grapheme_transcript in zip(audio_paths, transcripts, grapheme_transcripts):
                audio_path = audio_path.replace('txt', 'pcm')
                grpm_id_transcript = sentence_to_target(grapheme_transcript.split(), grpm2id)
                f.write(f'{audio_path}\t{transcript}\t{grpm_id_transcript}\n')


def generate_grapheme_labels(grapheme_transcripts, vocab_path: str):
    vocab_list = list()
    vocab_freq = list()

    for grapheme_transcript in grapheme_transcripts:
        graphemes = grapheme_transcript.split()
        for grapheme in graphemes:
            if grapheme not in vocab_list:
                vocab_list.append(grapheme)
                vocab_freq.append(1)
            else:
                vocab_freq[vocab_list.index(grapheme)] += 1

    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    vocab_dict = {
        'id': [0, 1, 2, 3],
        'grpm': ['<pad>', '<sos>', '<eos>', '<blank>'],
        'freq': [0, 0, 0, 0]
    }

    for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
        vocab_dict['id'].append(idx + 3)
        vocab_dict['grpm'].append(grpm)
        vocab_dict['freq'].append(freq)

    label_df = pd.DataFrame(vocab_dict)
    label_df.to_csv(vocab_path, encoding="utf-8", index=False)


def load_label(filepath):
    grpm2id = dict()
    id2grpm = dict()

    vocab_data_frame = pd.read_csv(filepath, encoding="utf-8")

    id_list = vocab_data_frame["id"]
    grpm_list = vocab_data_frame["grpm"]

    for _id, grpm in zip(id_list, grpm_list):
        grpm2id[grpm] = _id
        id2grpm[_id] = grpm
    return grpm2id, id2grpm


def sentence_to_target(transcript, grpm2id):
    target = str()

    for grpm in transcript:
        target += (str(grpm2id[grpm]) + ' ')

    return target[:-1]
