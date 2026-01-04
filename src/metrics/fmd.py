from typing import Union

from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.gaussian_estimators import GaussianEstimator
from frechet_music_distance.models import FeatureExtractor
from metrics.metrics import Metric


class FMD(Metric):
    """
    Frechet Music Distance
    """

    def __init__(
            self,
            logger,
            references_dir: str,
            sample_dir: str,
            extractor: Union[str, FeatureExtractor] = "clamp2",
            estimator: Union[str, GaussianEstimator] = "mle"
    ):
        super(FMD, self).__init__()
        self.logger = logger
        self.references_dir = references_dir
        self.sample_dir = sample_dir
        self.extractor = extractor
        self.estimator = estimator
        self.fmd = None

    def prepare(self):
        self.fmd = FrechetMusicDistance(
            feature_extractor=self.extractor,
            gaussian_estimator=self.estimator
        )

    def evaluate(self):
        fmd = self.fmd.score(self.references_dir, self.sample_dir)
        self.logger.log_metrics({"Frechet Music Distance": fmd})
