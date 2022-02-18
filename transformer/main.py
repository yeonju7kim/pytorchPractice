from data import *
from model import *

def main():
    tokenizer = get_tokenizer_dictionary()
    train_iter = get_train_iter()
    valid_iter = get_valid_iter()
    vocab = get_vocab(train_iter, tokenizer)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    history = history()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(transformer, optimizer, vocab, train_data, toknizer)
        val_loss = evaluate(transformer)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


if __name__ == '__main__':
    main()