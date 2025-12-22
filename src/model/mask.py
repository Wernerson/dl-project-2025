import random

import torch
import torch.nn.functional as F


class MaskingStrategy:

    def max_step(self, seq_len, dim) -> int:
        """
        Returns the maximum number of noise steps we can make.
        E.g., if we mask all tokens in a note per step, the maximum number of masking steps is seq_len.
        If we mask all tokens individually, the maximum number of steps is seq_len * dim. That's also the default.
        :param seq_len: sequence length
        :param dim: dimension of item in sequence (tokens per note)
        :return: maximum number of steps to noise/denoise
        """
        return seq_len * dim

    def noise_mask(self, x_0, t):
        """
        This mask masks all entries that should be masked before running denoise timestep t.
        This is only used during training since there we have a fully unmasked x_0.
        During inference, we only use the denoise_mask as we start with a fully masked x_t and then only denoise tokens of the denoise_mask.
        Note that not all true entries in this mask will be demasked at timestep t as the denoise_mask might differ.
        For that, see denoise_mask.
        :param x_0: the x_0 we will add noise to.
        :param t: timestep t
        :return: mask the dimension of x_0
        """
        raise NotImplementedError()

    def denoise_mask(self, x_t, t):
        """
        This mask marks which entries will be denoised at timestep t.
        Note that this might be fewer than the number of masked entries of x_t at timestep t.
        :param x_t: the x_t we will denoise.
        :param t: timestep t
        :return: mask of dimension x_t
        """
        raise NotImplementedError()


class NoteMasking(MaskingStrategy):
    """
    Masks t entire nodes randomly per step noise step.
    Then unmasks one entire random note during the denoising step.
    """

    def __init__(self, mask_token_id):
        super(NoteMasking, self).__init__()
        self.mask_token_id = mask_token_id

    def max_step(self, seq_len, dim) -> int:
        return seq_len

    def noise_mask(self, x_0, t):
        batch_size, seq_len, dim = x_0.shape
        rand_indices = torch.randperm(seq_len, device=x_0.device)
        mask_indices = rand_indices[:t]

        mask = torch.zeros(seq_len, dtype=torch.bool, device=x_0.device)
        mask[mask_indices] = True

        # Expand mask: [Batch, Seq, num tokens per note]
        batch_mask = mask.unsqueeze(1).expand(-1, dim).unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask

    def denoise_mask(self, x_t, t):
        batch_size, seq_len, dim = x_t.shape
        masked = x_t[0, :, 0] == self.mask_token_id
        idx = torch.arange(seq_len, device=x_t.device)
        unmask_idx = random.choice(idx[masked])

        mask = torch.zeros(seq_len, dtype=torch.bool, device=x_t.device)
        mask[unmask_idx] = True

        # Expand mask: [Batch, Seq, num tokens per note]
        batch_mask = mask.unsqueeze(1).expand(-1, dim).unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask


class SequentialNoteMasking(MaskingStrategy):
    """
    Unmasks all tokens from front to back.
    """

    def max_step(self, seq_len, dim) -> int:
        return seq_len

    def noise_mask(self, x_0, t):
        batch_size, seq_len, dim = x_0.shape
        mask = torch.zeros(seq_len, dtype=torch.bool, device=x_0.device)
        mask[seq_len - t:] = True

        # Expand mask: [Batch, Seq, num tokens per note]
        batch_mask = mask.unsqueeze(1).expand(-1, dim).unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask

    def denoise_mask(self, x_t, t):
        batch_size, seq_len, dim = x_t.shape
        mask = torch.zeros(seq_len, dtype=torch.bool, device=x_t.device)
        mask[seq_len - t] = True

        # Expand mask: [Batch, Seq, num tokens per note]
        batch_mask = mask.unsqueeze(1).expand(-1, dim).unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask


class ProbabilisticMasking(MaskingStrategy):
    """
    Unmasks certain tokens with higher probability.
    """

    def __init__(self, mask_token_id, samples_per_step, P_token, P_seq):
        super(ProbabilisticMasking, self).__init__()
        self.mask_token_id = mask_token_id
        self.samples_per_step = samples_per_step
        self.P_token = P_token
        self.P_seq = P_seq

    def max_step(self, seq_len, dim) -> int:
        return seq_len * dim // self.samples_per_step

    def extrapolated_sequence_prior(self, seq_len, device):
        L = len(self.P_seq)

        # target positions in [0, L0 - 1]
        pos = torch.linspace(0, L - 1, seq_len, device=device)

        # linear interpolation
        left = pos.floor().long()
        right = torch.clamp(left + 1, max=L - 1)
        w = pos - left.float()

        P_seq = torch.tensor(self.P_seq, device=device)
        scores = (1 - w) * P_seq[left] + w * P_seq[right]
        return F.softmax(scores, dim=0)

    def grid_logits(self, seq_len, device):
        seq_logits = self.extrapolated_sequence_prior(seq_len, device)
        token_logits = torch.tensor(self.P_token, device=device)
        logits = seq_logits[:, None] + token_logits[None, :]
        return logits

    def sample(self, x, logits, k):
        _, seq_len, dim = x.shape
        probs = F.softmax(logits.flatten(), dim=0)
        samples = torch.multinomial(probs, k)
        mask = torch.zeros(seq_len * dim, dtype=torch.bool, device=x.device)
        mask[samples] = True
        return mask.view(seq_len, dim)

    def noise_mask(self, x_0, t):
        batch_size, seq_len, dim = x_0.shape
        T = self.max_step(seq_len, dim)
        if t == T:
            return torch.ones((batch_size, seq_len, dim), dtype=torch.bool, device=x_0.device)
        logits = self.grid_logits(seq_len, x_0.device)
        mask = self.sample(x_0, logits, T - t)
        batch_mask = (~mask).unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask

    def denoise_mask(self, x_t, t):
        batch_size, seq_len, dim = x_t.shape

        logits = self.grid_logits(seq_len, x_t.device)
        masked = x_t[0] == self.mask_token_id
        logits[~masked] = float("-inf")

        mask = self.sample(x_t, logits, self.samples_per_step)
        batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return batch_mask


if __name__ == "__main__":
    P_token = [
        3,  # Pit
        5,  # Pos
        10,  # Bar
        1,  # Vel
        2,  # Dur
        15,  # Pro
        1,  # Tem
        1,  # Tim
    ]
    P_seq = [
        1,
        2,
        1
    ]
    mask_token = 77

    x = torch.randn(2, 10, 8)
    print(x.shape, x)

    t = 60
    ms = ProbabilisticMasking(mask_token, P_token, P_seq)
    noise_mask = ms.noise_mask(x, t)
    print(noise_mask.shape, noise_mask)
    print(x[noise_mask])

    x[noise_mask] = mask_token
    denoise_mask = ms.denoise_mask(x, t)
    print(denoise_mask.shape, denoise_mask)
    print(x[denoise_mask])
