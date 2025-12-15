import torch


class MaskingStrategy:

    def max_step(self, seq_len, dim) -> int:
        """
        Returns the maximum number of noise we can add.
        E.g., if we mask all tokens in a note per step, the maximum number of masking steps is seq_len.
        :param seq_len: sequence length
        :param dim: dimension of item in sequence (tokens per note)
        :return: maximum number of steps to noise/denoise
        """
        return seq_len * dim

    def mask(self, x, t):
        """
        Returns the mask with the correlating strategy for x_0 at denoise step t.
        :param x: the x_0 we want to denoise.
        :param t: timestep t
        :return: mask of dimension x_0
        """
        raise NotImplementedError()


class NoteMasking(MaskingStrategy):
    def __init__(self):
        super(NoteMasking, self).__init__()

    def max_step(self, seq_len, dim) -> int:
        return seq_len

    def mask(self, x, t):
        batch_size, seq_len, dim = x.shape
        rand_indices = torch.randperm(seq_len, device=x.device)
        mask_indices = rand_indices[:t]

        is_masked = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        is_masked[mask_indices] = True

        # Expand mask: [Batch, Seq, num tokens per note]
        is_masked_batch = is_masked.unsqueeze(1).expand(-1, dim).unsqueeze(0).expand(batch_size, -1, -1)
        return is_masked_batch
