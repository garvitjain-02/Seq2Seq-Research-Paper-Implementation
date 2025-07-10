import os
import csv
import sentencepiece as spm
from collections import Counter
from tqdm import tqdm

# --------------- CONFIGURATION ---------------
RAW_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\raw"
PROC_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\processed"
VOCAB_SIZE_SRC = 500
VOCAB_SIZE_TGT = 500
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '< SOS >'
EOS_TOKEN = '<EOS>'
SRC_LANG = 'en'
TGT_LANG = 'fr'
# Use only small CSVs and output to _small files
SPLITS = [
    ('train_small.csv', 'train_small.en.txt', 'train_small.fr.txt', 'train_small.proc.en', 'train_small.proc.fr'),
    ('val_small.csv', 'val_small.en.txt', 'val_small.fr.txt', 'val_small.proc.en', 'val_small.proc.fr'),
    ('test_small.csv', 'test_small.en.txt', 'test_small.fr.txt', 'test_small.proc.en', 'test_small.proc.fr'),
]
# ---------------------------------------------

def train_sentencepiece(input_txt, model_prefix, vocab_size):
    """Train SentencePiece model with proper argument handling for paths with spaces"""
    # Use dictionary format for arguments to handle spaces properly
    training_args = {
        'input': input_txt,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': 1.0,
        'model_type': 'unigram',
        'unk_id': 0,
        'pad_id': 1,
        'bos_id': 2,
        'eos_id': 3
    }
    
    spm.SentencePieceTrainer.Train(**training_args)

def load_sentencepiece(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
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
            # Add < SOS > and <EOS> to target
            tgt_tokens = [SOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
            src_out.write(' '.join(src_tokens) + '\n')
            tgt_out.write(' '.join(tgt_tokens) + '\n')

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    # Converting CSV to plain text (needed for SentencePiece training)
    print("Converting CSV files to text...")
    for csv_file, src_txt, tgt_txt, _, _ in SPLITS:
        csv_path = os.path.join(RAW_DIR, csv_file)
        src_txt_path = os.path.join(PROC_DIR, src_txt)
        tgt_txt_path = os.path.join(PROC_DIR, tgt_txt)
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        csv_to_txt(csv_path, src_txt_path, tgt_txt_path)
        print(f"Converted {csv_file} to text files")

    # Training SentencePiece models on train_small split
    print("Training SentencePiece models...")
    
    # Check if training files exist
    train_en_path = os.path.join(PROC_DIR, "train_small.en.txt")
    train_fr_path = os.path.join(PROC_DIR, "train_small.fr.txt")
    
    if not os.path.exists(train_en_path):
        raise FileNotFoundError(f"Training file not found: {train_en_path}")
    if not os.path.exists(train_fr_path):
        raise FileNotFoundError(f"Training file not found: {train_fr_path}")
    
    # Check if files are not empty
    if os.path.getsize(train_en_path) == 0:
        raise ValueError(f"Training file is empty: {train_en_path}")
    if os.path.getsize(train_fr_path) == 0:
        raise ValueError(f"Training file is empty: {train_fr_path}")
    
    try:
        train_sentencepiece(train_en_path, os.path.join(PROC_DIR, "spm_en_small"), VOCAB_SIZE_SRC)
        print("English SentencePiece model trained successfully")
    except Exception as e:
        print(f"Error training English SentencePiece model: {e}")
        raise
    
    try:
        train_sentencepiece(train_fr_path, os.path.join(PROC_DIR, "spm_fr_small"), VOCAB_SIZE_TGT)
        print("French SentencePiece model trained successfully")
    except Exception as e:
        print(f"Error training French SentencePiece model: {e}")
        raise

    # Loading SentencePiece models
    print("Loading SentencePiece models...")
    sp_src = load_sentencepiece(os.path.join(PROC_DIR, "spm_en_small"))
    sp_tgt = load_sentencepiece(os.path.join(PROC_DIR, "spm_fr_small"))

    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = build_vocab(sp_src, VOCAB_SIZE_SRC)
    tgt_vocab = build_vocab(sp_tgt, VOCAB_SIZE_TGT)
    src_vocab.add(UNK_TOKEN)
    tgt_vocab.update([UNK_TOKEN, SOS_TOKEN, EOS_TOKEN])

    # Process and save splits
    print("Processing and saving splits...")
    for csv_file, _, _, out_src, out_tgt in SPLITS:
        csv_path = os.path.join(RAW_DIR, csv_file)
        
        # Skip if CSV doesn't exist
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        out_src_path = os.path.join(PROC_DIR, out_src)
        out_tgt_path = os.path.join(PROC_DIR, out_tgt)
        process_and_save(csv_path, sp_src, sp_tgt, src_vocab, tgt_vocab, out_src_path, out_tgt_path)
        print(f"Processed {csv_file}")
    
    print("Done! Preprocessed files are in:", PROC_DIR)

if __name__ == '__main__':
    main()