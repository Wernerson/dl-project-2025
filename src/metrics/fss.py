import itertools
from pathlib import Path
from statistics import NormalDist

import note_seq.protobuf.music_pb2
import numpy as np
from note_seq import midi_file_to_note_sequence
from note_seq import quantize_note_sequence
from note_seq import sequences_lib

from metrics import Metric


class FSS(Metric):
    """
    Framewise Self-Similarity from SCHmUBERT
    """

    def __init__(self, reference_path, test_path):
        super(FSS, self).__init__()
        self.reference_path = reference_path
        self.test_path = test_path

    def evaluate(self):
        references = get_ns(self.reference_path)
        samples = get_ns(self.test_path)
        c, v = evaluate_consistency_variance(references, samples)
        return c, v


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
        return [0, 0]
    qns = quantize_note_sequence(ns, 4)
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


def get_OA_helper(values, prev_values):
    values, prev_values = np.array(values), np.array(prev_values)
    a = NormalDist(values.mean(), values.std())
    b = NormalDist(prev_values.mean(), prev_values.std())
    area = a.overlap(b)
    return area


def overlap(a, b):
    d1 = NormalDist(a.mean(), a.std() + 1e-6)
    d2 = NormalDist(b.mean(), b.std() + 1e-6)
    return d1.overlap(d2)


def get_disc_seq_with_overlap_fast(seq, prev_seq, target_overlap, lo=0, hi=90, speed=8):
    result = np.append(seq, prev_seq.copy())
    prev_window = np.append(prev_seq, seq)

    delta = 1 if prev_seq.mean() < (lo + hi) / 2 else -1
    delta *= speed
    while abs(delta) >= 1:
        if delta < 0:
            choices = np.where(result + delta >= lo)
        else:
            choices = np.where(result + delta < hi)

        if not len(choices) or not len(choices[0]) or choices[0].max() < len(seq):
            delta /= 2
            continue

        p = np.random.choice(choices[0][choices[0] >= len(seq)])

        result[p] += delta
        if (ovl := overlap(prev_window, result)) < target_overlap:
            result[p] -= delta
            delta /= 2

    return result[len(seq):], ovl


def get_bounded_seq_with_mu_sig(mu, sig, size, lo=0, hi=1):
    result = np.random.normal(mu, sig, size)

    high = np.where(result > hi)[0]
    print(result.mean())
    for h in high:
        d = result[h] - hi
        l = np.where(result < hi - d)[0]
        l = np.random.choice(l)
        result[h] = hi
        result[l] -= d

    return result


def construct_perfect_sample(mu_OA, var_OA, sample_size=1000, bars=64, width=4, hop=2):
    MIN_VAL = [0, 1]
    MAX_VAL = [90, 16]
    STEPS_PER_BAR = 16
    NOTES_PER_BAR = 8
    SECONDS_PER_STEP = 1 / 8
    wins = (bars - width) // hop

    samples = []
    prev_seq = [None, None]
    target_OAs = [get_bounded_seq_with_mu_sig(mu_OA[i], np.sqrt(var_OA[i]), sample_size) for i in [0, 1]]
    print([(target_OAs[i].mean(), target_OAs[i].var()) for i in [0, 1]])
    for k in range(sample_size):

        ns = note_seq.protobuf.music_pb2.NoteSequence()

        for j in range(2):
            seq = [np.random.randint(MIN_VAL[i], MAX_VAL[i], NOTES_PER_BAR * hop) for i in [0, 1]]
            prev_seq[j] = seq
            s = np.random.randint(0, NOTES_PER_BAR * hop)
            for p, d in zip(*seq):
                ns.notes.add(pitch=p, start_time=(s + np.random.randint(0, NOTES_PER_BAR * hop)) * SECONDS_PER_STEP,
                             end_time=(s + d) * SECONDS_PER_STEP, velocity=80)
        diffs = [0, 0]
        for w in range(wins):
            s = w * hop * STEPS_PER_BAR
            prev_seq[w % 2] = seq
            next_target_overlap = [target_OAs[i][k] - diffs[i] for i in [0, 1]]
            seq_overlaps = [
                get_disc_seq_with_overlap_fast(seq[i], prev_seq[(w + 1) % 2][i], next_target_overlap[i], MIN_VAL[i],
                                               MAX_VAL[i]) for i in [0, 1]]
            seq, overlaps = [[so[i] for so in seq_overlaps] for i in [0, 1]]

            diffs = [o - t[k] for o, t in zip(overlaps, target_OAs)]

            for p, d in zip(*seq):
                o = np.random.randint(0, NOTES_PER_BAR * hop)
                ns.notes.add(pitch=p, start_time=(s + o) * SECONDS_PER_STEP, end_time=(s + o + d) * SECONDS_PER_STEP,
                             velocity=80)
        samples.append(ns)
    return samples


def evaluate_consistency_variance(targets, preds):
    OA_t = [framewise_overlap_areas(t) for t in targets]
    OA_p = [framewise_overlap_areas(p) for p in preds]
    OA_t, OA_p = np.stack(OA_t), np.stack(OA_p)

    consistency = 1 - np.abs(OA_t.mean(0) - OA_p.mean(0)) / OA_t.mean(0)
    variance = 1 - np.abs(OA_t.var(0) - OA_p.var(0)) / OA_t.var(0)

    return np.clip(consistency, 0, 1), np.clip(variance, 0, 1)
