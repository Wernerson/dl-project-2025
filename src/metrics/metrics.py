from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def evaluate(self):
        raise NotImplemented
