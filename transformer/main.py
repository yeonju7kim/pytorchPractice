from data import *
from model import *
from utils.history_manage import *

def main():
    de_en_data = DeEnData()

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, len(de_en_data.vocab_transform[SRC_LANGUAGE]),
                                     len(de_en_data.vocab_transform[TGT_LANGUAGE]), FFN_HID_DIM)
    optimizer = optimizer_fn(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    history = ValueHistory()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = transformer.train_epoch(optimizer, de_en_data.train_dataLoader)
        val_loss = transformer.evaluate(de_en_data.valid_dataLoader)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        transformer.save(epoch, train_loss, val_loss)
        history.add_history('train_loss', train_loss)
        history.add_history('val_loss', val_loss)
    history.save_csv_all_history("transformer_history", "history")
    print(transformer.translate("Eine Gruppe von Menschen steht vor einem Iglu .", de_en_data.text_transform, de_en_data.vocab_transform))

if __name__ == '__main__':
    main()