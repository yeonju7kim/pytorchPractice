import torch
import torch.nn as nn

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
initialize_method = nn.init.xavier_uniform_

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)