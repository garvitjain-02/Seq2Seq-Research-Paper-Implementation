import torch
import sentencepiece as spm
from train_seq2seq import Encoder, Decoder, Seq2Seq, PAD_ID, SOS_ID, EOS_ID, DEVICE, EMB_DIM, HID_DIM

# Updated constants
PROC_DIR = r"C:\Users\Lenovo\Desktop\Research paper\Seq2Seq-Research-Paper-Implementation\data\processed"
VOCAB_SIZE_SRC = 500
VOCAB_SIZE_TGT = 500

# Load models
sp_src = spm.SentencePieceProcessor()
sp_src.Load(f"{PROC_DIR}/spm_en_small.model")
sp_tgt = spm.SentencePieceProcessor()
sp_tgt.Load(f"{PROC_DIR}/spm_fr_small.model")

encoder = Encoder(VOCAB_SIZE_SRC, EMB_DIM, HID_DIM).to(DEVICE)
decoder = Decoder(VOCAB_SIZE_TGT, EMB_DIM, HID_DIM).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
checkpoint = torch.load("models/seq2seq_model.pth", map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

def translate(sentence):
    model.eval()
    tokens = sp_src.Encode(sentence, out_type=int)
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    src_len = torch.tensor([len(tokens)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        hidden, cell = encoder(src_tensor, src_len)
        input = torch.tensor([SOS_ID], dtype=torch.long).to(DEVICE)
        result = []
        for _ in range(50):
            output, hidden, cell = decoder(input, hidden, cell)
            top1 = output.argmax(1)
            if top1.item() == EOS_ID:
                break
            result.append(top1.item())
            input = top1
    return sp_tgt.Decode(result)

def translate_debug(sentence):
    model.eval()
    tokens = sp_src.Encode(sentence, out_type=int)
    print("Input tokens:", tokens)
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    src_len = torch.tensor([len(tokens)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        hidden, cell = encoder(src_tensor, src_len)
        input = torch.tensor([SOS_ID], dtype=torch.long).to(DEVICE)
        result = []
        for _ in range(50):
            output, hidden, cell = decoder(input, hidden, cell)
            top1 = output.argmax(1)
            if top1.item() == EOS_ID:
                break
            result.append(top1.item())
            input = top1
    print("Output token IDs:", result)
    print("Output tokens:", [sp_tgt.IdToPiece(id) for id in result])

# Example usage
print(translate("Agreements in the commision"))
print(translate_debug("Agreements in the commision"))
