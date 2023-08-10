__version__ = "0.0.1"

from ._sample_data import tissuenet_sample
from .widgets._widget import (
    TrainWidget,
    model_config_widget,
    train_config_widget,
)

__all__ = (
    "tissuenet_sample",
    "train_config_widget",
    "model_config_widget",
    "TrainWidget",
)
