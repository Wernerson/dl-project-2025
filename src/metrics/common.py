from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Sequence

import muspy
import muspy.metrics as mpm
import numpy as np
from tqdm import tqdm


class Metric(ABC):
    @abstractmethod
    def evaluate(self):
        raise NotImplemented

    def prepare(self):
        pass


MUSPY_METRICS = {  # TODO adjust parameters
    "Drum in Pattern Rate": partial(mpm.drum_in_pattern_rate, meter="duple"),
    "Drum Pattern Consistency": mpm.drum_pattern_consistency,
    "Empty Beat Rate": mpm.empty_beat_rate,
    "Empty Measure Rate": partial(mpm.empty_measure_rate, measure_resolution=1),
    "Groove Consistency": partial(mpm.groove_consistency, measure_resolution=1),
    "Num of Pitch Classes Used": mpm.n_pitch_classes_used,
    "Num of Pitches Used": mpm.n_pitches_used,
    "Pitch Class Entropy": mpm.pitch_class_entropy,
    "Pitch Entropy": mpm.pitch_entropy,
    "Pitch in Scale Rate": partial(mpm.pitch_in_scale_rate, root=2, mode="major"),
    "Pitch Range": mpm.pitch_range,
    "Polyphony": mpm.polyphony,
    "Polyphony Rate": mpm.polyphony_rate,
    "Scale Consistency": mpm.scale_consistency,
}


class CommonMetrics(Metric):
    def __init__(
            self,
            references_dir: str,
            sample_dir: str,
            metrics: Sequence[str] = MUSPY_METRICS.keys()
    ):
        super(CommonMetrics, self).__init__()
        self.references_dir = references_dir
        self.sample_dir = sample_dir
        self.metrics = metrics

    def evaluate(self):
        metrics = {}
        for metric in tqdm(self.metrics, desc="Calculating metrics"):
            ref_value = get_stat(metric, self.references_dir)
            if ref_value is not None:
                metrics[f"Mean Reference {metric}"] = ref_value

            sample_value = get_stat(metric, self.sample_dir)
            if sample_value is not None:
                metrics[f"Mean Sample {metric}"] = sample_value

        return metrics


def get_stat(metric, dir):
    func = MUSPY_METRICS[metric]
    files = list(Path(dir).glob("**/*.mid"))
    values = []
    for file in tqdm(files, f"Files in {dir}", leave=False):
        m = muspy.read(file, kind="midi")
        values.append(func(m))
    if len(values) == 0:
        return None
    value = np.mean(values)
    return value.item() if not np.isnan(value) else None
