from dataclasses import dataclass


@dataclass
class SpeechModeType:
    PHOENTIC: str = 'phonetic'
    SPELLING: str = 'spelling'


@dataclass
class KsponSpeechVocabType:
    CHARACTER: str = 'character'
    GRAPHEME: str = 'grapheme'
    # SUBWORD: str = 'subword'
