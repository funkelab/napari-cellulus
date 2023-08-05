import numpy as np

from pathlib import Path

TISSUENET_SAMPLE = Path(__file__).parent / "sample_data/tissuenet-sample.npy"


def tissuenet_sample():
    (x, y) = np.load(TISSUENET_SAMPLE, "r")
    x = x.transpose(0, 3, 1, 2)
    y = y.transpose(0, 3, 1, 2).astype(np.uint8)
    return [
        (
            x[100:105],
            {
                "name": "Raw",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "image",
        ),
        (
            y[100:105],
            {
                "name": "Labels",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "Labels",
        ),
    ]
