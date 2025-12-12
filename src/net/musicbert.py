import torch
import torch.nn as nn

class MusicBERT(nn.Module):
    def __init__(self,
                 vocab_size=2048, 
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 max_seq_len=1024,
                 tokens_per_note=8
                 ): 
                
        super().__init__()

        self.num_tokens_per_note = tokens_per_note

        # 1. The Embeddings (The "CP" Layer)
        # We share one embedding layer for all domains (using offsets)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # CP Downsampling: tokens_per_note tokens -> 1 vector
        # Input: [Batch, SeqLen*tokens_per_note, Dim] -> Concat tokens_per_note -> [Batch, SeqLen, Dim*tokens_per_note]
        # Linear: [Dim*tokens_per_note] -> [Dim]
        self.downsample = nn.Linear(hidden_dim * self.num_tokens_per_note, hidden_dim)

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
        # Linear: [Dim] -> [Dim*tokens_per_note]
        self.upsample = nn.Linear(hidden_dim, hidden_dim * self.num_tokens_per_note)

        # Prediction Head: [Dim] -> [VocabSize]
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, padding_mask=None):
        """
        x: [Batch, SeqLen * tokens_per_note] (Flattened & Offset Indices)
        """
        batch_size, flat_len = x.shape
        seq_len = flat_len // self.num_tokens_per_note

        # --- A. Embedding & Grouping ---
        # 1. Embed individual tokens
        x_emb = self.embedding(x)  # [Batch, SeqLen*tokens_per_note, Dim]

        # 2. Group into notes (CP Logic)
        # View as [Batch, SeqLen, tokens_per_note, Dim] -> Flatten last 2 dims -> [Batch, SeqLen, tokens_per_note*Dim]
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
        x_tfm = self.transformer(x_down, src_key_padding_mask=padding_mask)

        # --- D. Upsampling & Prediction ---
        # 1. Expand back to tokens_per_note vectors per note
        # [Batch, SeqLen, Dim] -> [Batch, SeqLen, tokens_per_note*Dim]
        x_up = self.upsample(x_tfm)

        # 2. Reshape to flattened sequence again for prediction
        # [Batch, SeqLen, tokens_per_note*Dim] -> [Batch, SeqLen, tokens_per_note, Dim] -> [Batch, SeqLen*tokens_per_note, Dim]
        x_up_seq = x_up.view(batch_size, seq_len, self.num_tokens_per_note, -1).view(batch_size, flat_len, -1)

        # 3. Project to Vocab
        logits = self.output_head(x_up_seq)  # [Batch, SeqLen*tokens_per_note, Vocab]

        return logits





