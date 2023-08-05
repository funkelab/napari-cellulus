from .gp.nodes.napari_image_source import NapariImageSource

import math
from typing import Tuple

import gunpowder as gp
from torch.utils.data import IterableDataset

import numpy as np
from napari.layers import Image


class NapariDataset(IterableDataset):  # type: ignore
    def __init__(
        self,
        layer: Image,
        crop_size: Tuple[int, ...],
        control_point_spacing: int,
        control_point_jitter: float,
    ):
        """A dataset that serves random samples from a zarr container.

        Args:

            data:

                A numpy array to use.
                The data should have shape `(s, c, [t,] [z,] y, x)`, where
                `s` = # of samples, `c` = # of channels, `t` = # of frames, and
                `z`/`y`/`x` are spatial extents. The dataset should have an
                `"axis_names"` attribute that contains the names of the used
                axes, e.g., `["s", "c", "y", "x"]` for a 2D dataset.


            crop_size:

                The size of data crops used during training (distinct from the
                "patch size" of the method: from each crop, multiple patches
                will be randomly selected and the loss computed on them). This
                should be equal to the input size of the model that predicts
                the OCEs.

            control_point_spacing:

                The distance in pixels between control points used for elastic
                deformation of the raw data.

            control_point_jitter:

                How much to jitter the control points for elastic deformation
                of the raw data, given as the standard deviation of a normal
                distribution with zero mean.
        """

        self.layer = layer
        self.crop_size = crop_size
        self.control_point_spacing = control_point_spacing
        self.control_point_jitter = control_point_jitter
        self.__read_meta_data()

        assert len(crop_size) == self.num_spatial_dims, (
            f'"crop_size" must have the same dimension as the '
            f'spatial(temporal) dimensions of the "{self.dataset_config.dataset_name}" '
            f"dataset which is {self.num_spatial_dims}, but it is {crop_size}"
        )

        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")

        self.pipeline = (
            NapariImageSource(self.layer, self.raw)
            + gp.RandomLocation()
            + gp.ElasticAugment(
                control_point_spacing=(self.control_point_spacing,)
                * self.num_spatial_dims,
                jitter_sigma=(self.control_point_jitter,)
                * self.num_spatial_dims,
                rotation_interval=(0, math.pi / 2),
                scale_interval=(0.9, 1.1),
                subsample=4,
                spatial_dims=self.num_spatial_dims,
            )
            # + gp.SimpleAugment(mirror_only=spatial_dims, transpose_only=spatial_dims)
        )

    def __yield_sample(self):
        """An infinite generator of crops."""

        with gp.build(self.pipeline):
            while True:
                # request one sample, all channels, plus crop dimensions
                request = gp.BatchRequest()
                request[self.raw] = gp.ArraySpec(
                    roi=gp.Roi(
                        (0,) * self.num_dims,
                        (1, self.num_channels, *self.crop_size),
                    )
                )

                sample = self.pipeline.request_batch(request)
                yield sample[self.raw].data[0]

    def __read_meta_data(self):
        self.num_dims = len(self.layer.data.shape)
        self.num_spatial_dims = self.num_dims - 2
        self.num_channels = self.layer.data.shape[1]
        self.num_samples = self.layer.data.shape[0]
        self.sample_dim = 0
        self.channel_dim = 1
        self.time_dim = None

    def get_num_channels(self):
        return self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims
