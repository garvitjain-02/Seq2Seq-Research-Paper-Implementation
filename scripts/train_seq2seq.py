import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu

# Configuration
PROC_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\processed"
SRC_MODEL = os.path.join(PROC_DIR, "spm_en_small.model")
TGT_MODEL = os.path.join(PROC_DIR, "spm_fr_small.model")
BATCH_SIZE = 16
VOCAB_SIZE_SRC = 500
VOCAB_SIZE_TGT = 500
EMB_DIM = 256
HID_DIM = 512
N_EPOCHS = 5
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset & DataLoader 
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, sp_src, sp_tgt):
        self.src_sentences = self.read_file_with_fallback(src_file)
        self.tgt_sentences = self.read_file_with_fallback(tgt_file)
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        assert len(self.src_sentences) == len(self.tgt_sentences)

    def read_file_with_fallback(self, path):
        for enc in ['utf-8-sig', 'utf-8', 'utf-16', 'latin1']:
            try:
                with open(path, encoding=enc) as f:
                    return f.read().strip().split('\n')
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode {path} with tried encodings.")

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = self.src_sentences[idx].split()
        tgt_tokens = self.tgt_sentences[idx].split()
        src_ids = [self.sp_src.piece_to_id(token) for token in src_tokens]
        tgt_ids = [self.sp_tgt.piece_to_id(token) for token in tgt_tokens]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = list(src_batch)
    tgt_batch = list(tgt_batch)
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt_batch], dtype=torch.long)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID, batch_first=True)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_ID, batch_first=True)
    return src_pad, tgt_pad, src_lens, tgt_lens

# Encoder-Decoder Model Classes
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input = tgt[:, 0]  # <SOS>

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

if __name__ == "__main__":
    # Loading the SentencePiece models
    sp_src = spm.SentencePieceProcessor()
    sp_src.Load(SRC_MODEL)
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.Load(TGT_MODEL)

    # Loading all the datasets
    train_en_path = os.path.join(PROC_DIR, "train_small.proc.en")
    train_fr_path = os.path.join(PROC_DIR, "train_small.proc.fr")
    val_en_path = os.path.join(PROC_DIR, "val_small.proc.en")
    val_fr_path = os.path.join(PROC_DIR, "val_small.proc.fr")

    train_dataset = TranslationDataset(
        train_en_path,
        train_fr_path,
        sp_src, sp_tgt
    )
    val_dataset = TranslationDataset(
        val_en_path,
        val_fr_path,
        sp_src, sp_tgt
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Building the model
    encoder = Encoder(VOCAB_SIZE_SRC, EMB_DIM, HID_DIM).to(DEVICE)
    decoder = Decoder(VOCAB_SIZE_TGT, EMB_DIM, HID_DIM).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Training loop
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            print(f"src shape: {src.shape}, tgt shape: {tgt.shape}, src_len: {src_len}")
            optimizer.zero_grad()
            output = model(src, src_len, tgt)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{N_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        bleu_scores = []
        with torch.no_grad():
            for src, tgt, src_len, tgt_len in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                output = model(src, src_len, tgt, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                val_loss += loss.item()
                pred_tokens = output.argmax(-1).cpu().numpy().reshape(tgt.shape[0], -1)
                tgt_tokens = tgt.cpu().numpy().reshape(tgt.shape[0], -1)
                for pred, ref in zip(pred_tokens, tgt_tokens):
                    # Remove PAD, EOS, SOS tokens as needed
                    pred_seq = [tok for tok in pred if tok not in [PAD_ID, EOS_ID, SOS_ID]]
                    ref_seq = [tok for tok in ref if tok not in [PAD_ID, EOS_ID, SOS_ID]]
                    bleu_scores.append(sentence_bleu([ref_seq], pred_seq))
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"Validation BLEU Score: {avg_bleu:.4f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': N_EPOCHS
    }, "models/seq2seq_model.pth")
    print("Model saved to models/seq2seq_model.pth")
