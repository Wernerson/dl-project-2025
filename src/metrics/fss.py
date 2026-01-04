import itertools
from pathlib import Path
from statistics import NormalDist

import numpy as np
from metrics.metrics import Metric
from note_seq import midi_file_to_note_sequence
from note_seq import quantize_note_sequence
from note_seq import sequences_lib


class FSS(Metric):
    """
    Framewise Self-Similarity from SCHmUBERT
    """

    def __init__(self, logger, references_dir, sample_dir):
        super(FSS, self).__init__()
        self.logger = logger
        self.references_dir = references_dir
        self.sample_dir = sample_dir

    def evaluate(self):
        references = get_ns(self.references_dir)
        samples = get_ns(self.sample_dir)
        (c_pitch, c_duration), (v_pitch, v_duration) = evaluate_consistency_variance(references, samples)
        self.logger.log_metrics({
            "Pitch Consistency": c_pitch,
            "Pitch Variance": v_duration,
            "Duration Consistency": c_duration,
            "Duration Variance": v_pitch,
        })


def get_ns(path):
    files = list(Path(path).glob("**/*.mid"))
    ns = [midi_file_to_note_sequence(file) for file in files]
    return ns


#############################################################################
# Below copied from https://github.com/plassma/symbolic-music-discrete-diffusion/blob/master/utils/eval_utils.py
#############################################################################

def frame_statistics(bars):
    bars = list(itertools.chain(*bars))
    stats = lambda x: NormalDist(np.mean(x), np.std(x) + 1e-6) if len(x) else NormalDist(1, 1e-6)
    return stats([n.pitch for n in bars]), stats([n.quantized_end_step - n.quantized_start_step for n in bars])


def framewise_overlap_areas(ns, width=4, hop=2):
    if not len(ns.notes):
        return None
    try:
        qns = quantize_note_sequence(ns, 4)
    except:
        return None
    steps_per_bar = sequences_lib.steps_per_bar_in_quantized_sequence(qns)
    assert steps_per_bar == 16.
    steps_per_bar = 16

    by_bar = [[] for _ in range(max([n.quantized_end_step for n in qns.notes]) // steps_per_bar + 1)]

    for note in qns.notes:
        k = note.quantized_start_step // steps_per_bar
        by_bar[k].append(note)
        # if note.quantized_end_step // steps_per_bar != k:#todo: how 2 handle bar crossing notes?
        #    by_bar[note.quantized_end_step // steps_per_bar].append(note)

    frames = []
    for f in range((len(by_bar) - width) // hop + 1):
        start_bar = hop * f
        frames.append(frame_statistics(by_bar[start_bar:start_bar + width]))

    OAs = []
    for i in range(len(frames) - 1):
        OAs.append([frames[i][j].overlap(frames[i + 1][j]) for j in [0, 1]])

    return np.array(OAs).mean(0)


def evaluate_consistency_variance(targets, preds):
    OA_t = list(filter(lambda x: isinstance(x, np.ndarray), [framewise_overlap_areas(t) for t in targets]))
    OA_p = list(filter(lambda x: isinstance(x, np.ndarray), [framewise_overlap_areas(p) for p in preds]))
    OA_t, OA_p = np.stack(OA_t), np.stack(OA_p)
    consistency = np.clip(1 - np.abs(OA_t.mean(0) - OA_p.mean(0)) / OA_t.mean(0), 0, 1)
    variance = np.clip(1 - np.abs(OA_t.var(0) - OA_p.var(0)) / OA_t.var(0), 0, 1)
    return consistency, variance
