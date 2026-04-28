import pytest

from lesionshiftai.data.labels import BENIGN, HAM_CLASS_COLUMNS, HAM_MALIGNANT_CLASSES, MALIGNANT


pytestmark = pytest.mark.unit


def test_label_constants() -> None:
    assert BENIGN == 0
    assert MALIGNANT == 1
    assert len(HAM_CLASS_COLUMNS) == 7
    assert {"MEL", "BCC", "AKIEC"}.issubset(HAM_MALIGNANT_CLASSES)
