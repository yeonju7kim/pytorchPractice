from torch.utils.data import DataLoader
from copy import copy, deepcopy
from config import *
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import *

class DeEnData():
    def __init__(self):
        self.train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        self.valid_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        self.test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

        self._set_preprocess()

        def collate_fn(batch):
            print(batch)
            src_batch, tgt_batch = [], []
            for src_sample, tgt_sample in batch:
                print(src_sample)
                src_batch.append(self.text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
                tgt_batch.append(self.text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
            src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
            tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
            return src_batch, tgt_batch

        self.train_dataLoader = DataLoader(self.train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        print([i for i in self.train_dataLoader])
        self.valid_dataLoader = DataLoader(self.valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        self.test_dataLoader = DataLoader(self.test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)



    def _set_preprocess(self):
        def yield_tokens(data_iter: Iterable, language) -> List[str]:
            language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

            for data_sample in data_iter:
                yield self.token_transform[language](data_sample[language_index[language]])

        def tensor_transform(token_ids: List[int]):
            return torch.cat((torch.tensor([BOS_IDX]),
                              torch.tensor(token_ids),
                              torch.tensor([EOS_IDX])))

        def sequential_transforms(*transforms):
            def func(txt_input):
                for transform in transforms:
                    txt_input = transform(txt_input)
                return txt_input
            return func

        self.token_transform = {}
        self.token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language=f'{SRC_LANGUAGE}_core_news_sm')
        self.token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language=f'{TGT_LANGUAGE}_core_web_sm')

        self.vocab_transform = {}
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            self.vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(deepcopy(self.train_iter), ln),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)
            self.vocab_transform[ln].set_default_index(UNK_IDX)

        self.text_transform = {}
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            self.text_transform[ln] = sequential_transforms(self.token_transform[ln],  # 토큰화(Tokenization)
                                                       self.vocab_transform[ln],  # 수치화(Numericalization)
                                                       tensor_transform)


