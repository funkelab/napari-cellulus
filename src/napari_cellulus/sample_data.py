from pathlib import Path

import numpy as np

TISSUE_NET_SAMPLE = (
    Path(__file__).parent / "sample_data/tissue_net_skin_sample.npz"
)


def tissue_net_sample():
    data = np.load(TISSUE_NET_SAMPLE)
    raw, gt = data["raw"], data["gt"]

    num_samples = raw.shape[0]
    indices = np.random.randint(0, num_samples - 1, 10)
    raw = raw[indices]
    gt = gt[indices]
    return [
        (
            raw,
            {
                "name": "Raw",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "image",
        ),
        (
            gt,
            {
                "name": "Labels",
                "metadata": {"axes": ["s", "c", "y", "x"]},
            },
            "Labels",
        ),
    ]
