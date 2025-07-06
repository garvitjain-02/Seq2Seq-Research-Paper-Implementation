import os
import csv
import sentencepiece as spm
from collections import Counter
from tqdm import tqdm

# --------------- CONFIGURATION ---------------
RAW_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\raw"
PROC_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\processed"
VOCAB_SIZE_SRC = 6000
VOCAB_SIZE_TGT = 6000
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
SRC_LANG = 'en'
TGT_LANG = 'fr'
# ---------------------------------------------

def train_sentencepiece(input_txt, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram',
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3
    )

def load_sentencepiece(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def csv_to_txt(csv_path, src_txt_path, tgt_txt_path):
    with open(csv_path, encoding='utf-8') as f, \
         open(src_txt_path, 'w', encoding='utf-8') as src_f, \
         open(tgt_txt_path, 'w', encoding='utf-8') as tgt_f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            src, tgt = row[0].strip(), row[1].strip()
            src_f.write(src + '\n')
            tgt_f.write(tgt + '\n')

def build_vocab(sp_model, vocab_size):
    return set(sp_model.id_to_piece(i) for i in range(vocab_size))

def process_and_save(csv_path, src_sp, tgt_sp, src_vocab, tgt_vocab, out_src_path, out_tgt_path):
    with open(csv_path, encoding='utf-8') as f, \
         open(out_src_path, 'w', encoding='utf-8') as src_out, \
         open(out_tgt_path, 'w', encoding='utf-8') as tgt_out:
        reader = csv.reader(f)
        for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_path)}"):
            if len(row) < 2:
                continue
            src_sent, tgt_sent = row[0].strip(), row[1].strip()
            src_tokens = src_sp.encode(src_sent, out_type=str)
            tgt_tokens = tgt_sp.encode(tgt_sent, out_type=str)
            # Replace OOVs with <UNK>
            src_tokens = [w if w in src_vocab else UNK_TOKEN for w in src_tokens]
            tgt_tokens = [w if w in tgt_vocab else UNK_TOKEN for w in tgt_tokens]
            # Reverse source tokens
            src_tokens = src_tokens[::-1]
            # Add <SOS> and <EOS> to target
            tgt_tokens = [SOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
            src_out.write(' '.join(src_tokens) + '\n')
            tgt_out.write(' '.join(tgt_tokens) + '\n')

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    # Converting CSV to plain text (needed for SentencePiece training)
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(RAW_DIR, f"{split}.csv")
        src_txt = os.path.join(PROC_DIR, f"{split}.en.txt")
        tgt_txt = os.path.join(PROC_DIR, f"{split}.fr.txt")
        csv_to_txt(csv_path, src_txt, tgt_txt)


    # Training SentencePiece models on train split
    print("Training SentencePiece models...")
    train_sentencepiece(os.path.join(PROC_DIR, "train.en.txt"), os.path.join(PROC_DIR, "spm_en"), VOCAB_SIZE_SRC)
    train_sentencepiece(os.path.join(PROC_DIR, "train.fr.txt"), os.path.join(PROC_DIR, "spm_fr"), VOCAB_SIZE_TGT)

    # Loading SentencePiece models
    sp_src = load_sentencepiece(os.path.join(PROC_DIR, "spm_en"))
    sp_tgt = load_sentencepiece(os.path.join(PROC_DIR, "spm_fr"))

    # Build vocabularies
    src_vocab = build_vocab(sp_src, VOCAB_SIZE_SRC)
    tgt_vocab = build_vocab(sp_tgt, VOCAB_SIZE_TGT)
    src_vocab.add(UNK_TOKEN)
    tgt_vocab.update([UNK_TOKEN, SOS_TOKEN, EOS_TOKEN])

    # Process and save splits
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(RAW_DIR, f"{split}.csv")
        out_src = os.path.join(PROC_DIR, f"{split}.proc.en")
        out_tgt = os.path.join(PROC_DIR, f"{split}.proc.fr")
        process_and_save(csv_path, sp_src, sp_tgt, src_vocab, tgt_vocab, out_src, out_tgt)
    print("Done! Preprocessed files are in:", PROC_DIR)

if __name__ == '__main__':
    main()
