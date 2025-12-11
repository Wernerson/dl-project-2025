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


        # --- 1. Create the Constraint Mask in Init ---
        vocab_size = net.output_head.out_features
        
        # Start with everything invalid (-inf)
        mask = torch.full((len(offsets), vocab_size), float('-inf'))
        
        # Open up the valid ranges
        bounds = offsets + [self.mask_token_id] 
        for i in range(len(offsets)):
            # Set valid range to 0.0
            mask[i, bounds[i]:bounds[i+1]] = 0.0
            
        # Register as buffer (handles device movement automatically)
        self.register_buffer('constraint_mask', mask)


    def compute_logits(self, x_flat, apply_constraints=False):
        """
        Shared logic for Training and Inference.
        Applies Network -> Reshape -> Constraints.
        Args:
            x_flat: [Batch, Seq*8] (Global Indices)
        Returns:
            logits: [Batch, Seq, 8, Vocab] (Constrained)
        """
        # 1. Network Pass
        logits = self.net(x_flat) 
        
        # 2. Reshape to [Batch, Seq, 8, Vocab]
        # (Handling the case where MusicBERT returns flat [B, S*8, V])
        if logits.dim() == 3:
            batch_size = logits.size(0)
            logits = logits.view(batch_size, -1, len(self.offsets), logits.size(-1))
            
        # 3. Apply Constraints (Conditional)
        if apply_constraints:
            logits = logits + self.constraint_mask.unsqueeze(0).unsqueeze(0)
        
        return logits
    

    def forward(self, x_structured):
        """
        User Interface: Accepts Raw MIDI (0-127).
        """
        # 1. Apply Offsets
        x_offset = x_structured + self.offsets
        
        # 2. Flatten
        x_flat = x_offset.view(x_structured.size(0), -1)
        
        # 3. Compute (Net + Constraints)
        return self.compute_logits(x_flat, apply_constraints=False)
    

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

        logits = self.compute_logits(x_t_flat, apply_constraints=False)

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
        tokens_per_note = self.offsets.shape[0] # 8
        
        # Sampling Hyperparameters
        top_k = 50       # Restrict to top 50 choices (prevents weird bad notes)
        temperature = 1.0 # 1.0 = standard, >1.0 = chaotic, <1.0 = conservative
        
        # 1. Start: All Masked
        x_t = torch.full((1, n_notes, tokens_per_note), self.mask_token_id, 
                         device=self.device, dtype=torch.long)
        
        # 2. Random Permutation
        random_order = torch.randperm(n_notes, device=self.device)
        
        # 3. Iterative Unmasking
        for step in range(n_notes):
            target_idx = random_order[step]
            
            # A. Network Prediction
            logits = self.compute_logits(x_t.view(1, -1), apply_constraints=True)
            
            # Focus on the specific note we are updating
            # Shape: [1, 1, 8, Vocab]
            note_logits = logits[:, target_idx, :, :].unsqueeze(1)


            # 1. Apply Temperature
            # Higher T makes distribution flatter (more random)
            note_logits = note_logits / temperature
            
            # 2. Top-K Filtering
            # We filter independently for each of the 8 attributes
            # Get top k values
            v, i = torch.topk(note_logits, top_k)
            
            # Create a mask of -inf for everything NOT in the top k
            # (Everything smaller than the k-th best value becomes -inf)
            min_values = v[:, :, :, -1].unsqueeze(-1).expand_as(note_logits)
            note_logits = torch.where(note_logits < min_values, 
                                      torch.tensor(float('-inf'), device=self.device), 
                                      note_logits)

            # 3. Sample from the filtered distribution
            probs = F.softmax(note_logits, dim=-1)
            
            # We flatten to 2D [8, Vocab] to use multinomial, then reshape back
            # Result: [1, 1, 8] containing the chosen token IDs
            sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, 1, tokens_per_note)
                        
            # B. Update
            x_t[0, target_idx, :] = sampled_ids[0, 0, :]
            
        return torch.relu(x_t - self.offsets)
    
    
    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters())