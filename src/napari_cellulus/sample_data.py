from pathlib import Path

import numpy as np

FLUO_C2DL_HUH7_SAMPLE = (
    Path(__file__).parent / "sample_data/Fluo-C2DL-Huh7-sample.npy"
)


def fluo_c2dl_huh7_sample():
    raw = np.load(FLUO_C2DL_HUH7_SAMPLE)
    num_samples = raw.shape[0]
    indices = np.random.choice(np.arange(num_samples), 5, replace=False)
    raw = raw[indices]

    return [
        (
            raw,
            {
                "name": "Raw",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "image",
        )
    ]
