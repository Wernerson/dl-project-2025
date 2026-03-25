import random

import lightning as L
import torch
import torch.nn.functional as F
from mask_vis import plot_mask_vis


class MusicBertDiffusion(L.LightningModule):
    def __init__(
            self,
            net,
            optimizer, lr_scheduler,
            offsets, mask_strategy,
            denoise_mask_loss=False,
            denoise_mask_fw=True,
            error_correct = False
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mask_strategy = mask_strategy
        self.denoise_mask_loss = denoise_mask_loss
        self.denoise_mask_fw = denoise_mask_fw
        self.error_correct = error_correct

        # Register offsets buffer
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))

        self.mask_token_id = net.output_head.out_features - 1

        # Global Special IDs (0=BOS, 1=PAD, 2=EOS, 3=UNK)
        self.pad_token_id = 1

        # --- 1. Create the Constraint Mask in Init ---
        vocab_size = net.output_head.out_features

        # Start with everything invalid (-inf)
        mask = torch.full((len(offsets), vocab_size), float('-inf'))

        # Open up the valid ranges
        bounds = offsets + [self.mask_token_id]
        for i in range(len(offsets)):
            # Set valid range to 0.0
            mask[i, bounds[i]:bounds[i + 1]] = 0.0

        # Register as buffer (handles device movement automatically)
        self.register_buffer('constraint_mask', mask)

        self.tokens_per_note = self.offsets.shape[0]

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

        # 0. Detect Padding Rows (Where ALL attributes are 0)
        # Shape: [Batch, Seq]
        is_padding = (x_structured == 0).all(dim=-1)

        # 1. Apply Offsets
        x_offset = x_structured + self.offsets

        # 2. Flatten
        x_flat = x_offset.view(x_structured.size(0), -1)

        # Expand is_padding to match flattened shape: [Batch, Seq] -> [Batch, Seq*tokens_per_note]
        is_padding_flat = is_padding.unsqueeze(-1).expand(-1, -1, self.tokens_per_note).reshape(x_structured.size(0),
                                                                                                -1)

        # Override: Where padding is True, set to PAD_ID
        x_flat = torch.where(is_padding_flat,
                             torch.tensor(self.pad_token_id, device=self.device),
                             x_flat)

        # 3. Compute (Net + Constraints)
        return self.compute_logits(x_flat, apply_constraints=False, padding_mask=is_padding)

    def _sample_forward_loss(self, x_0):
        batch_size, seq_len, num_attribs = x_0.shape

        # Create Mask
        T = self.mask_strategy.max_step(seq_len, num_attribs)
        t = random.randint(1, T)
        noise_mask = self.mask_strategy.noise_mask(x_0, t)

        # Detect Padding
        padding = x_0 == 0
        # Ensure we NEVER mask padding (Padding must stay visible as Padding)
        noise_mask = noise_mask & (~padding)

        # Apply Offsets
        x_offset = x_0 + self.offsets

        # Override Padding with PAD_ID (1)
        # (Otherwise 0+offset becomes a valid note like Pitch 4)
        x_offset = torch.where(padding,
                               torch.tensor(self.pad_token_id, device=self.device),
                               x_offset)

        if self.error_correct:
            # === NEW GIDD APPROACH ===
            prob_noise = 0.1
            rand_tensor = torch.rand_like(x_0.float())

            # 1. Identify Uniform Noise Targets
            # (Not masked, Not padding, but selected for noise)
            uniform_mask = (rand_tensor < prob_noise) & (~noise_mask) & (~padding)

            # 2. Generate Structure-Preserving Random Tokens
            # Bounds = [Offset_0, Offset_1, ..., MaskTokenID]
            bounds = torch.cat([self.offsets, torch.tensor([self.mask_token_id], device=self.device)])
            valid_random_tokens = torch.zeros_like(x_offset)

            for i in range(num_attribs):
                low = bounds[i]
                high = bounds[i + 1]
                valid_random_tokens[:, :, i] = torch.randint(
                    low, high, (batch_size, seq_len), device=self.device
                )

            # 3. Apply Corruption (Masking + Uniform Noise)
            x_masked = x_offset.clone()

            # Apply Masks
            x_masked = torch.where(noise_mask,
                                   torch.tensor(self.mask_token_id, device=self.device),
                                   x_masked)
            # Apply Noise
            x_masked = torch.where(uniform_mask, valid_random_tokens, x_masked)

            # 4. Compute Logits
            x_t_flat = x_masked.view(batch_size, -1)
            logits = self.compute_logits(x_t_flat, apply_constraints=False)

            # 5. Compute Loss
            # GIDD Target: Predict everything that was corrupted (Masked OR Noised)

            if self.denoise_mask_loss:
                # If using strategy-specific targets, add uniform targets to them
                denoise_mask = self.mask_strategy.denoise_mask(x_masked, t)
                final_target_mask = denoise_mask | uniform_mask
            else:
                # Standard: predict all corruptions
                final_target_mask = noise_mask | uniform_mask

            final_target_mask = final_target_mask & (~padding)

            masked_logits = logits[final_target_mask]
            masked_targets = x_offset[final_target_mask]

            if masked_targets.numel() == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return F.cross_entropy(masked_logits, masked_targets)

        else:
            # Mask
            x_masked = torch.where(noise_mask,
                                   torch.tensor(self.mask_token_id, device=self.device),
                                   x_offset)
            # flatten
            x_t_flat = x_masked.view(batch_size, -1)

            # compute logits
            logits = self.compute_logits(x_t_flat, apply_constraints=False)

            if self.denoise_mask_loss:
                # get denoise mask, these are the relevant tokens we want to predict at timestamp t
                denoise_mask = self.mask_strategy.denoise_mask(x_masked, t)
                denoise_mask = denoise_mask & (~padding)

                # Loss
                masked_logits = logits[denoise_mask]
                masked_targets = x_offset[denoise_mask]
            else:
                masked_logits = logits[noise_mask]
                masked_targets = x_offset[noise_mask]

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

    def sample(self, seq_len=128, top_k = 50, temperature = 1.0, batch_size=1):
        # top_k: restrict to top X choices (prevents weird bad notes)
        # temperature: 1.0 = standard, >1.0 = chaotic, <1.0 = conservative
        dim = self.tokens_per_note

        # 1. Start: All Masked
        x_t = torch.full((batch_size, seq_len, dim), self.mask_token_id, device=self.device, dtype=torch.long)

        # 2. Iterative Unmasking
        T = self.mask_strategy.max_step(seq_len, dim)
        for t in reversed(range(1, T + 1)):
            # Mask x_t
            mask = self.mask_strategy.denoise_mask(x_t, t)
            x_t[mask] = self.mask_token_id

            # A. Network Prediction [batch_size, seq len * tokens per note * Vocab]
            logits = self.compute_logits(x_t.view(batch_size, -1), apply_constraints=True)

            # 1. Apply Temperature
            # Higher T makes distribution flatter (more random)
            logits = logits / temperature

            # 2. Top-K Filtering
            # We filter independently for each of the 8 attributes
            # Get top k values
            v, _ = torch.topk(logits, top_k)

            # Create a mask of -inf for everything NOT in the top k
            # (Everything smaller than the k-th best value becomes -inf)
            min_values = v[:, :, :, -1].unsqueeze(-1)
            logits[logits < min_values] = float('-inf')

            # 3. Sample from the filtered distribution [1, seq len, tokens per note, Vocab]
            probs = F.softmax(logits, dim=-1)

            # We flatten to 2D [x, Vocab] to use multinomial, then reshape back
            # Result: [batch_size, seq_len, 8] containing the chosen token IDs
            sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len, dim)

            # B. Update only masked entries
            if self.denoise_mask_fw:
                x_t[mask] = sampled_ids[mask]
            else:
                x_t = sampled_ids

        return x_t - self.offsets

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.lr_scheduler(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # Update every step, not every epoch
            }
        }

    def self_correct(self, x_generated, iterations=5, threshold_remove=0.1, threshold_add=0.9):
        """
        GIDD Self-Correction Step.
        Args:
            x_generated: output from sample() [Batch, Seq, 8] (without offsets)
        """
        # Add offsets back (internal model needs global indices)
        x_curr = x_generated + self.offsets

        stats = {
            "total_corrections": 0,
            "corrections_per_step": []
        }

        for i in range(iterations):
            # 1. Compute Logits
            x_flat = x_curr.view(x_curr.size(0), -1)
            logits = self.compute_logits(x_flat, apply_constraints=True)
            probs = F.softmax(logits, dim=-1)

            # 2. Get Confidence
            # Current tokens
            curr_indices = x_curr.unsqueeze(-1)
            curr_confidence = probs.gather(-1, curr_indices).squeeze(-1)

            # Best tokens
            best_confidence, best_tokens = probs.max(dim=-1)

            # 1. Don't touch existing PADs
            is_existing_pad = (x_curr == self.pad_token_id)

            # 2. Don't create NEW PADs (The cause of your error)
            is_new_pad = (best_tokens == self.pad_token_id)

            # Decision Rule:
            should_swap = (curr_confidence < threshold_remove) & \
                          (best_confidence > threshold_add) & \
                          (~is_existing_pad) & \
                          (~is_new_pad)

            # 4. Count and Swap
            num_swaps = should_swap.sum().item()
            stats["total_corrections"] += num_swaps
            stats["corrections_per_step"].append(num_swaps)

            x_curr = torch.where(should_swap, best_tokens, x_curr)

            if num_swaps == 0:
                break

        return x_curr - self.offsets, stats