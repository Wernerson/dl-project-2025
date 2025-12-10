#import torch.nn as nn
#
#
#class MusicBERT(nn.Module):
#
#    def __init__(self, input_size, lin1_size, lin2_size, lin3_size, output_size):
#        super(MusicBERT, self).__init__()
#        self.model = nn.Sequential(
#            nn.Linear(input_size, lin1_size),
#            nn.ReLU(),
#            nn.Linear(lin1_size, lin2_size),
#            nn.ReLU(),
#            nn.Linear(lin2_size, lin3_size),
#            nn.ReLU(),
#            nn.Linear(lin3_size, output_size),
#        )
#
#    def forward(self, x):
#        return self.model(x)


import torch
import torch.nn as nn
from transformers import BertConfig, BertEncoder  # Standard HF modules (or simple PyTorch ones)


class MusicBERT(nn.Module):
    def __init__(self,
                 vocab_size=2048, #TODO: check correct vocab size to allow for mask tokens and everything else potentially necessary
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 max_seq_len=1024, #TODO: probably signifficantly smaller for now, is that an issue
                 ): 
                
        super().__init__()

        # 1. The Embeddings (The "CP" Layer)
        # We share one embedding layer for all domains (using offsets)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # CP Downsampling: 8 tokens -> 1 vector
        # Input: [Batch, SeqLen*8, Dim] -> Concat 8 -> [Batch, SeqLen, Dim*8]
        # Linear: [Dim*8] -> [Dim]
        self.downsample = nn.Linear(hidden_dim * 8, hidden_dim)

        # 2. The Transformer Backbone
        # You can use standard PyTorch TransformerEncoder or HF BertEncoder
        # Using PyTorch native for zero dependencies:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-Norm usually trains better
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Position Embeddings (Standard Learned)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # 3. The Output Heads (Upsampling)
        # Linear: [Dim] -> [Dim*8]
        self.upsample = nn.Linear(hidden_dim, hidden_dim * 8)

        # Prediction Head: [Dim] -> [VocabSize]
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: [Batch, SeqLen * 8] (Flattened & Offset Indices)
        """
        batch_size, flat_len = x.shape
        seq_len = flat_len // 8

        # --- A. Embedding & Grouping ---
        # 1. Embed individual tokens
        x_emb = self.embedding(x)  # [Batch, SeqLen*8, Dim]

        # 2. Group into notes (CP Logic)
        # View as [Batch, SeqLen, 8, Dim] -> Flatten last 2 dims -> [Batch, SeqLen, 8*Dim]
        x_grouped = x_emb.view(batch_size, seq_len, -1)

        # 3. Downsample to single vector per note
        x_down = self.downsample(x_grouped)  # [Batch, SeqLen, Dim]

        # --- B. Add Position Info ---
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_down = x_down + self.pos_emb(positions)
        x_down = self.layer_norm(x_down)
        x_down = self.dropout(x_down)

        # --- C. Transformer Processing ---
        # [Batch, SeqLen, Dim]
        x_tfm = self.transformer(x_down)

        # --- D. Upsampling & Prediction ---
        # 1. Expand back to 8 vectors per note
        # [Batch, SeqLen, Dim] -> [Batch, SeqLen, 8*Dim]
        x_up = self.upsample(x_tfm)

        # 2. Reshape to flattened sequence again for prediction
        # [Batch, SeqLen, 8*Dim] -> [Batch, SeqLen, 8, Dim] -> [Batch, SeqLen*8, Dim]
        x_up_seq = x_up.view(batch_size, seq_len, 8, -1).view(batch_size, flat_len, -1)

        # 3. Project to Vocab
        logits = self.output_head(x_up_seq)  # [Batch, SeqLen*8, Vocab]

        # 4. (Optional) Reshape to structured if you prefer
        # return logits.view(batch_size, seq_len, 8, -1)

        return logits

    @staticmethod
    def loss(x_0, x_0_hat, mask, padding):
        """
        batch_len, seq_len, dim = x_0_hat.shape
        mask = mask & ~padding
        y = x_0.clone()
        y[~padding] -= 1
        y = y.reshape(batch_len, seq_len)
        y = F.one_hot(y, dim).float()
        mask = mask.repeat_interleave(4, dim=-1)
        if ~mask.any():
            return 0.  # useful was masked
        x = x_0_hat[mask]
        y = y[mask]
        loss = 0.
        for xi, yi in zip(x, y):
            loss = loss + F.cross_entropy(xi, yi)
        return loss
        """

        # compare x_0 and x_0_hat only on the positions where we have a mask token

    def predict(self, x_t):
        """
        batch_len, seq_len, dim = x_t.shape
        x = self.forward(x_t)
        x = x.argmax(dim=-1) + 1
        x = x.reshape(batch_len, -1, self.in_dim)
        return x
        """

        # transform input to tokens
        # call forward to get the sequence of logits

        # mask out invalid tokens / set the logits to -inf

        # perform argmax over the sequence of logits to get tokens
        # call TokensToOctupleMidi(x_t_hat) with the predictions
        # return correctly formated octuple Midi

        pass

    def OctupleMidiToTokens(x_t, mask):
        # transform to scalar sequence
        # add domain offset
        # mask token ?

        pass

    def TokensToOctupleMidi(x_t_hat):
        # input is a sequence of tokens from full vocabulary
        # transform back to full vector sequence
        # maybe perform checks if valid here

        pass
