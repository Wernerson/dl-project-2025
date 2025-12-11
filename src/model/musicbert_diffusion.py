import random
import torch
import torch.nn.functional as F
import lightning as L

class MusicBertDiffusion(L.LightningModule):
    def __init__(self, net, optimizer, offsets):
        super().__init__()
        self.net = net
        self.optimizer_cls = optimizer
        
        # Register offsets buffer
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        
        self.mask_token_id = net.output_head.out_features - 1 

    def forward(self, x_structured):
        x_offset = x_structured + self.offsets
        x_flat = x_offset.view(x_structured.size(0), -1)
        return self.net(x_flat)

    def _sample_forward_loss(self, x_0):
        batch_size, seq_len, num_attribs = x_0.shape

        # 1. Create Mask
        t = random.randint(1, seq_len)
        rand_indices = torch.randperm(seq_len, device=self.device)
        mask_indices = rand_indices[:t]
        
        is_masked = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        is_masked[mask_indices] = True
        
        # Expand mask: [Batch, Seq, 4]
        is_masked_batch = is_masked.unsqueeze(1).expand(-1, num_attribs).unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Apply Offsets & Masking
        x_offset = x_0 + self.offsets
        
        x_t_flat = torch.where(is_masked_batch, 
                               torch.tensor(self.mask_token_id, device=self.device), 
                               x_offset).view(batch_size, -1)

        # 3. Pass to Net
        logits = self.net(x_t_flat) 

        # 4. Reshape Logits
        logits = logits.view(batch_size, seq_len, num_attribs, -1)

        # 5. Loss
        target = x_offset
        masked_logits = logits[is_masked_batch] 
        masked_targets = target[is_masked_batch]
        
        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return F.cross_entropy(masked_logits, masked_targets)

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"] if isinstance(batch, dict) else batch
        loss = self._sample_forward_loss(x)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"] if isinstance(batch, dict) else batch
        loss = self._sample_forward_loss(x)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["input_ids"] if isinstance(batch, dict) else batch
        loss = self._sample_forward_loss(x)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        n_notes = 128 
        tokens_per_note = self.offsets.shape[0] # Should be 8
        
        # 1. Start: All Masked
        x_t = torch.full((1, n_notes, tokens_per_note), self.mask_token_id, 
                         device=self.device, dtype=torch.long)
        
        # 2. Generate Random Order
        # We create a random permutation of indices [0, 1, ... 127]
        # This defines the "Random Path" we will take to generate the song.
        # This guarantees we visit every note exactly once without repeating.
        random_order = torch.randperm(n_notes, device=self.device)
        
        # 3. Iterative Unmasking
        for step in range(n_notes):
            # Get the index of the single note we want to reveal this step
            target_idx = random_order[step]
            
            # A. Predict
            # The model sees the current state (partially filled, partially masked)
            logits = self.net(x_t.view(1, -1)) 
            logits = logits.view(1, n_notes, tokens_per_note, -1)
            
            # B. Greedy Selection (Argmax)
            predictions = torch.argmax(logits, dim=-1) # [1, N, 8]
            
            # C. Update ONLY the specific note we selected
            # We copy the predicted values for 'target_idx' into x_t
            x_t[0, target_idx, :] = predictions[0, target_idx, :]
            
        # 4. Finish
        # Remove offsets to return to MIDI values (0-127)
        return torch.relu(x_t - self.offsets)
    
    
    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters())