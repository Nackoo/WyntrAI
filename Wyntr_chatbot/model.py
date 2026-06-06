import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.1, nhead: int = 8, dim_feedforward: int = 0):
        super().__init__()
        d_model = embed_dim
        if dim_feedforward <= 0:
            dim_feedforward = 4 * d_model

        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.1, nhead: int = 8, dim_feedforward: int = 0):
        super().__init__()
        d_model = embed_dim
        self.vocab_size = vocab_size
        if dim_feedforward <= 0:
            dim_feedforward = 4 * d_model

        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        output = self.transformer_decoder(
            x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, sos_idx: int, eos_idx: int, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    @staticmethod
    def _padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        return seq == pad_idx

    def forward(self, src, trg):
        src_pad_mask = self._padding_mask(src, self.pad_idx)
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)

        trg_in = trg[:, :-1]
        trg_len = trg_in.size(1)
        tgt_mask = self._causal_mask(trg_len, src.device)
        tgt_pad_mask = self._padding_mask(trg_in, self.pad_idx)

        logits = self.decoder(
            trg_in, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return logits

    @torch.no_grad()
    def generate(self, src_tensor, max_len: int = 40, temperature: float = 1.0, beam_width: int = 1):
        self.eval()
        src_pad_mask = self._padding_mask(src_tensor, self.pad_idx)
        memory = self.encoder(src_tensor, src_key_padding_mask=src_pad_mask)

        if beam_width <= 1:
            return self._greedy_generate(memory, src_pad_mask, max_len, temperature)
        return self._beam_generate(memory, src_pad_mask, max_len, beam_width)

    def _greedy_generate(self, memory, src_pad_mask, max_len, temperature):
        device = memory.device
        tokens = [self.sos_idx]
        output_tokens = []

        for _ in range(max_len):
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = self._causal_mask(tgt.size(1), device)
            logits = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask)
            next_logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = next_logits.argmax(dim=-1).item()
            else:
                next_token = self._sample_token(next_logits, temperature, top_p=0.92)

            if next_token == self.eos_idx:
                break
            tokens.append(next_token)
            output_tokens.append(next_token)

        return output_tokens

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float, top_p: float = 0.92) -> int:
        logits = logits / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cum_probs - sorted_probs) < top_p
        sorted_probs = sorted_probs * cutoff.float()

        if sorted_probs.sum() == 0:
            sorted_probs = torch.ones_like(sorted_probs)

        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled = torch.multinomial(sorted_probs, 1).item()
        return sorted_idx[sampled].item()

    def _beam_generate(self, memory, src_pad_mask, max_len, beam_width, length_penalty: float = 0.7):
        device = memory.device
        beams = [(0.0, [self.sos_idx])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, tokens in beams:
                if tokens[-1] == self.eos_idx:
                    seq = tokens[1:-1]
                    norm_score = score / max(len(seq), 1) ** length_penalty
                    completed.append((norm_score, seq))
                    continue

                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_mask = self._causal_mask(tgt.size(1), device)
                logits = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                top_probs, top_idxs = log_probs.topk(beam_width)

                for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
                    candidates.append((score + prob, tokens + [idx]))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        for score, tokens in beams:
            seq = tokens[1:]
            if seq and seq[-1] == self.eos_idx:
                seq = seq[:-1]
            norm_score = score / max(len(seq), 1) ** length_penalty
            completed.append((norm_score, seq))

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        return []
