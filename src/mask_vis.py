import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm

from src.model.mask import ProbabilisticMasking


def plot_mask_vis(mask_vis, T):
    fig = plt.figure()
    plt.title("Masking Schedule")
    cmap = plt.get_cmap("viridis", T)
    norm = BoundaryNorm(range(T + 1), cmap.N)
    plt.imshow(mask_vis.T, cmap=cmap, norm=norm, aspect="auto")
    cbar = plt.colorbar(orientation="horizontal", label="Demasking Timestep")
    cbar.set_ticks([0, T - 1])
    cbar.set_ticklabels(["Late (t=0)", "Early (t=T)"])
    plt.yticks(range(8), labels=[
        "Pit",
        "Pos",
        "Bar",
        "Vel",
        "Dur",
        "Pro",
        "Tem",
        "Tim",
    ])  # todo verify that this order is correct...
    plt.xlabel("Note Sequence")
    plt.close(fig)
    return fig


P_token = [  # probability per token in note, will be normalized (8 numbers)
    2,  # Pit
    3,  # Pos
    3,  # Bar
    1,  # Vel
    2,  # Dur
    3,  # Pro
    1,  # Tem
    1,  # Tim
]
P_seq = [  # probability distribution in sequence, arbitrary length
    3,
    1,
    1,
]


if __name__ == "__main__":
    mask_token_id = 99
    # ms = NoteMasking(mask_token_id)
    # ms = SequentialNoteMasking()
    ms = ProbabilisticMasking(mask_token_id, 8, P_token, P_seq)

    seq_len, dim = 128, 8
    T = ms.max_step(seq_len, dim)

    mask_vis = torch.zeros((seq_len, dim))
    x_t = torch.full((1, seq_len, dim), mask_token_id, dtype=torch.long)

    for t in reversed(range(1, T + 1)):
        mask = ms.denoise_mask(x_t, t)
        mask_vis[mask_vis > 0] += 1
        mask_vis[mask[0]] = 1
        x_t[mask] = 1  # "denoise"

    fig = plot_mask_vis(mask_vis, T)
    fig.show()