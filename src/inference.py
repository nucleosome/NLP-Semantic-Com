import json
import torch
from src.transceiver import DeepSC
from utils import SNR_to_noise, greedy_decode, SeqtoText

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepSCInference:
    def __init__(self, checkpoint_path: str, vocab_path: str, channel: str = "AWGN", snr: float = 10.0):
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.start_idx = self.vocab.get("<START>", 1)
        self.end_idx = self.vocab.get("<END>", 2)
        self.channel = channel
        self.n_var = SNR_to_noise(snr)

        self.model = DeepSC(
            num_layers=4,
            src_vocab_size=self.vocab_size,
            trg_vocab_size=self.vocab_size,
            src_max_len=30,
            trg_max_len=30,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.seq2text = SeqtoText(self.vocab, self.end_idx)

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = [self.vocab.get(w, self.vocab.get("<UNK>", 3)) for w in text.lower().split()]
        tokens = [self.start_idx] + tokens + [self.end_idx]
        return torch.tensor([tokens], dtype=torch.long).to(device)

    def predict(self, text: str, max_len: int = 30) -> str:
        src = self._tokenize(text)
        with torch.no_grad():
            output = greedy_decode(
                self.model, src, self.n_var, max_len,
                self.pad_idx, self.start_idx, self.channel
            )
        return self.seq2text.sequence_to_text(output[0].tolist())
