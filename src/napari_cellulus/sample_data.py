from pathlib import Path

import numpy as np
import tifffile

TISSUE_NET_SAMPLE = Path(__file__).parent / "sample_data/tissue_net_sample.npy"
FLUO_N2DL_HELA = Path(__file__).parent / "sample_data/fluo_n2dl_hela.tif"


def fluo_n2dl_hela_sample():
    x = tifffile.imread(FLUO_N2DL_HELA)
    return [
        (
            x,
            {
                "name": "Raw",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "image",
        )
    ]


def tissue_net_sample():
    (x, y) = np.load(TISSUE_NET_SAMPLE, "r")
    x = x.transpose(0, 3, 1, 2)
    y = y.transpose(0, 3, 1, 2).astype(np.uint8)
    return [
        (
            x,
            {
                "name": "Raw",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "image",
        ),
        (
            y,
            {
                "name": "Labels",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "Labels",
        ),
    ]
