from dataclasses import dataclass
from typing import Tuple, List

_REPO_ID = 'deepghs/insightface'
_DEFAULT_MODEL = 'buffalo_l'


@dataclass
class Face:
    bbox: Tuple[float, float, float, float]
    det_score: float
    keypoints: List[Tuple[float, float]]
    gender: str = None
    age: int = None

    def to_det_tuple(self) -> Tuple[Tuple[float, float, float, float], str, float]:
        return self.bbox, 'face', self.det_score
