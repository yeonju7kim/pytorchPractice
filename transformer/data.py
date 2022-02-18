from config import *
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from torchtext.data.utils import get_tokenizer

def get_tokenizer_dictionary():
    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language=f'{SRC_LANGUAGE}_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language=f'{TGT_LANGUAGE}_core_web_sm')
    return token_transform

def get_train_iter():
    return Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

def get_valid_iter():
    return Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

def get_test_iter():
    return Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

def yield_tokens(data_iter: Iterable, tokenizer) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield tokenizer[SRC_LANGUAGE](data_sample[language_index[SRC_LANGUAGE]])

def get_vocab(train_iter, tokenizer):
    vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
        train_iter = train_iter
    return vocab_transform
