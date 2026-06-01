import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Reads the input sequence and compresses it into a context vector.
    Each word is looked up in an embedding table, then processed by an LSTM.
    The final hidden + cell state is the "meaning" of the input sentence.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len)  token indices
        embedded = self.dropout(self.embedding(src))          # (batch, seq_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, seq_len, hidden)  — all timestep outputs
        # hidden / cell: (num_layers, batch, hidden)
        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Bahdanau-style additive attention.
    At each decoder step, the decoder looks back at every encoder output
    and decides how much to attend to each position.
    This fixes the bottleneck of a single context vector for long inputs.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v    = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  (batch, hidden)  — top layer of decoder
        # encoder_outputs: (batch, src_len, hidden)
        src_len = encoder_outputs.size(1)
        hidden  = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden)
        energy  = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # (batch, src_len, hidden)
        scores  = self.v(energy).squeeze(2)                           # (batch, src_len)
        weights = torch.softmax(scores, dim=1)                        # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)    # (batch, 1, hidden)
        return context.squeeze(1), weights                            # (batch, hidden), (batch, src_len)


class Decoder(nn.Module):
    """
    Generates the response one token at a time.
    At each step it receives:
      - the previous token (or <SOS> at step 0)
      - its own previous hidden/cell state
      - an attention-weighted summary of the encoder outputs
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention  = Attention(hidden_size)
        # input to LSTM = embedding + context vector
        self.lstm = nn.LSTM(
            embed_dim + hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out  = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, token, hidden, cell, encoder_outputs):
        """Single decoding step. Returns logits, new hidden, new cell."""
        # token: (batch,)
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))        # (batch, 1, embed_dim)
        # attention uses only the top LSTM layer hidden state
        top_hidden = hidden[-1]                                             # (batch, hidden)
        context, attn_weights = self.attention(top_hidden, encoder_outputs)
        context_expanded = context.unsqueeze(1)                            # (batch, 1, hidden)
        lstm_input = torch.cat([embedded, context_expanded], dim=2)        # (batch, 1, embed+hidden)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden)
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context], dim=1)
        )                                                                   # (batch, vocab_size)
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """
    Full encoder-decoder model.

    Training  : uses teacher forcing (feed the correct previous token).
    Inference : generates autoregressively; supports temperature + beam search.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, sos_idx: int, eos_idx: int, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        """
        src : (batch, src_len)
        trg : (batch, trg_len)   — includes <SOS> at position 0
        Returns logits of shape (batch, trg_len-1, vocab_size)
        """
        batch_size  = src.size(0)
        trg_len     = trg.size(1)
        vocab_size  = self.decoder.vocab_size

        enc_outputs, hidden, cell = self.encoder(src)

        outputs = torch.zeros(batch_size, trg_len - 1, vocab_size, device=src.device)
        token   = trg[:, 0]   # <SOS>

        for t in range(1, trg_len):
            logits, hidden, cell, _ = self.decoder.forward_step(token, hidden, cell, enc_outputs)
            outputs[:, t - 1] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            token = trg[:, t] if use_teacher else logits.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def generate(self, src_tensor, max_len: int = 40, temperature: float = 0.8, beam_width: int = 1):
        """
        Generate a response for a single input tensor (1, src_len).
        Returns list of token indices (without SOS/EOS).
        """
        self.eval()
        enc_outputs, hidden, cell = self.encoder(src_tensor)

        if beam_width <= 1:
            return self._greedy_generate(enc_outputs, hidden, cell, max_len, temperature)
        else:
            return self._beam_generate(enc_outputs, hidden, cell, max_len, beam_width)

    def _greedy_generate(self, enc_outputs, hidden, cell, max_len, temperature):
        token  = torch.tensor([self.sos_idx], device=enc_outputs.device)
        tokens = []
        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder.forward_step(token, hidden, cell, enc_outputs)
            if temperature == 0.0:
                next_token = logits.argmax(dim=1)
            else:
                probs      = torch.softmax(logits / temperature, dim=1)
                next_token = torch.multinomial(probs, 1).squeeze(1)
            idx = next_token.item()
            if idx == self.eos_idx:
                break
            tokens.append(idx)
            token = next_token
        return tokens

    def _beam_generate(self, enc_outputs, hidden, cell, max_len, beam_width):
        """Simple beam search."""
        # each beam: (score, token_list, hidden, cell)
        beams = [(0.0, [], hidden, cell)]
        completed = []

        token = torch.tensor([self.sos_idx], device=enc_outputs.device)

        for _ in range(max_len):
            candidates = []
            for score, tokens, h, c in beams:
                if tokens and tokens[-1] == self.eos_idx:
                    completed.append((score, tokens[:-1]))
                    continue
                last_token = torch.tensor(
                    [tokens[-1] if tokens else self.sos_idx],
                    device=enc_outputs.device
                )
                logits, nh, nc, _ = self.decoder.forward_step(last_token, h, c, enc_outputs)
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                top_probs, top_idxs = log_probs.topk(beam_width)
                for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
                    candidates.append((score + prob, tokens + [idx], nh, nc))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        return beams[0][1] if beams else []