"""labels.py

Benign and malignant integer values, HAM10000 class columns.
"""
from typing import Final, List, Set


BENIGN: Final[int] = 0
MALIGNANT: Final[int] = 1

HAM_CLASS_COLUMNS: Final[List[str]] = [
    "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"
]
HAM_MALIGNANT_CLASSES: Final[Set[str]] = {
    "MEL", "BCC", "AKIEC"
}
