from dataclasses import dataclass
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


@dataclass
class KsponSpeechModeType:
    PHOENTIC: str = 'phonetic'
    SPELLING: str = 'spelling'


@dataclass
class KsponSpeechVocabType:
    CHARACTER: str = 'character'
    SUBWORD: str = 'subword'
    GRAPHEME: str = 'grapheme'


class KsponSpeech:
    train_trn = ('train.trn',)
    eval_trn = ("eval_clean.trn", "eval_other.trn")

    def preprocess(
            self,
            dataset_path: str,
            script_file_dir: str,
            mode: str = KsponSpeechModeType.PHOENTIC,
            unit: KsponSpeechVocabType = KsponSpeechVocabType.SUBWORD,
    ):

        train_audio_paths, train_transcripts = self.preprocess_sentence(script_file_dir, self.train_trn, mode)
        eval_audio_paths, eval_transcripts = self.preprocess_sentence(script_file_dir, self.eval_trn, mode)

        audio_paths = train_audio_paths + eval_audio_paths
        transcripts = train_transcripts + eval_transcripts

        manifest_file_path: str = './mani.csv'
        vocab_path: str = './vocab.csv'
        self.save_manifest(
            audio_paths,
            transcripts,
            manifest_file_path,
            vocab_path,
            unit=KsponSpeechVocabType.CHARACTER
        )

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
            with open(os.path.join(script_file_dir, script_name), 'r') as f:
                for line in tqdm(f.readlines()):
                    audio_path, raw_transcript = line.split(" :: ")
                    audio_paths.append(audio_path)

                    file_name = os.path.basename(audio_path)
                    replace = PERCENT_FILES.get(file_name[12:18], None)
                    transcript = sentence_filter(raw_transcript, mode=mode, replace=replace)
                    transcripts.append(transcript)

        return audio_paths, transcripts

    def save_manifest(
            self,
            audio_paths: list,
            transcripts: list,
            manifest_file_path: str,
            vocab_path: str,
            unit=KsponSpeechVocabType.CHARACTER
    ):
        vocabs = []
        if unit == KsponSpeechVocabType.GRAPHEME:
            vocabs = self.generate_grapheme(transcripts, vocab_path)
        elif unit == KsponSpeechVocabType.CHARACTER:
            vocabs = self.generate_character(transcripts, vocab_path)

        vocab2id, id2vocab = self.get_label(vocab_path)
        with open(manifest_file_path, "w") as f:
            for audio_path, transcript, vocab in zip(audio_paths, transcripts, vocabs):
                vocab_id_transcript = self.sentence_to_target(vocab, vocab2id)
                f.write(f'{audio_path}\t{transcript}\t{vocab_id_transcript}\n')

    def generate_vocab(
            self,
            sentences: list,
            vocab_path: str,
            vocab_type=KsponSpeechVocabType.GRAPHEME
    ) -> list:
        vocabs = list()
        vocab_freq = list()
        result = list()
        for sentence in tqdm(sentences):
            sentence_vocab = sentence
            if vocab_type == KsponSpeechVocabType.GRAPHEME:
                sentence_vocab = unicodedata.normalize('NFKD', sentence).replace(' ', '|').upper()

            result.append(sentence_vocab)
            for vocab in sentence_vocab:
                if vocab not in vocabs:
                    vocabs.append(vocab)
                    vocab_freq.append(1)
                else:
                    vocab_freq[vocabs.index(vocab)] += 1

        vocab_dict = {
            'id': [0, 1, 2],
            'vocab': ['<pad>', '<sos>', '<eos>'],
            'freq': [0, 0, 0]
        }

        for idx, (freq, grapheme) in enumerate(sorted(zip(vocab_freq, vocabs), reverse=True), start=3):
            vocab_dict['id'].append(idx)
            vocab_dict['vocab'].append(grapheme)
            vocab_dict['freq'].append(freq)

        if vocab_type == KsponSpeechVocabType.CHARACTER:
            vocab_dict['id'] = vocab_dict['id'][:2000]
            vocab_dict['vocab'] = vocab_dict['vocab'][:2000]
            vocab_dict['freq'] = vocab_dict['freq'][:2000]

        vocab_df = pd.DataFrame(vocab_dict)
        vocab_df.to_csv(vocab_path, encoding="utf-8", index=False)
        return result

    def generate_grapheme(self, sentences: list, vocab_path: str) -> list:
        vocabs = list()
        vocab_freq = list()
        graphemes = []
        for sentence in tqdm(sentences):
            sentence_grapheme = unicodedata.normalize('NFKD', sentence).replace(' ', '|').upper()
            graphemes.append(sentence_grapheme)
            for grapheme in sentence_grapheme:
                if grapheme not in vocabs:
                    vocabs.append(grapheme)
                    vocab_freq.append(1)
                else:
                    vocab_freq[vocabs.index(grapheme)] += 1

        vocab_dict = {
            'id': [0, 1, 2],
            'vocab': ['<pad>', '<sos>', '<eos>'],
            'freq': [0, 0, 0]
        }

        # vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocabs), reverse=True))
        for idx, (freq, grapheme) in enumerate(sorted(zip(vocab_freq, vocabs), reverse=True), start=3):
            vocab_dict['id'].append(idx)
            vocab_dict['vocab'].append(grapheme)
            vocab_dict['freq'].append(freq)

        vocab_df = pd.DataFrame(vocab_dict)
        vocab_df.to_csv(vocab_path, encoding="utf-8", index=False)
        return graphemes

    def generate_character(self, sentences: list, vocab_path: str) -> list:
        vocabs = list()
        vocab_freq = list()

        for sentence in sentences:
            for ch in sentence:
                if ch not in vocabs:
                    vocabs.append(ch)
                    vocab_freq.append(1)
                else:
                    vocab_freq[vocabs.index(ch)] += 1

        # sort together Using zip
        label_freq, label_list = zip(*sorted(zip(vocab_freq, vocabs), reverse=True))
        label = {
            'id': [0, 1, 2, 3],
            'vocab': ['<pad>', '<sos>', '<eos>', '<blank>'],
            'freq': [0, 0, 0, 0]
        }

        for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
            label['id'].append(idx + 4)
            label['vocab'].append(ch)
            label['freq'].append(freq)

        label['id'] = label['id'][:2000]
        label['vocab'] = label['vocab'][:2000]
        label['freq'] = label['freq'][:2000]

        label_df = pd.DataFrame(label)
        label_df.to_csv(vocab_path, encoding="utf-8", index=False)
        return sentences

    def generate_subword(self, sentences: list, vocab_path: str) -> list:
        vocabs = list()

    def get_label(self, vocab_path: str):
        vocab_data_frame = pd.read_csv(vocab_path, encoding="utf-8")
        id_list = vocab_data_frame["id"]
        vocab_list = vocab_data_frame["vocab"]

        vocab2id = dict()
        id2grpm = dict()
        for _id, grpm in zip(id_list, vocab_list):
            vocab2id[grpm] = _id
            id2grpm[_id] = grpm
        return vocab2id, id2grpm

    def sentence_to_target(self, transcript, vocab2id):
        target = str()
        for vocab in transcript:
            target += (str(vocab2id[vocab]) + ' ')

        return target[:-1]
