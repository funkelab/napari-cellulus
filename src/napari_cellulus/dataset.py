from typing import List

import gunpowder as gp
import numpy as np
from napari.layers import Image
from torch.utils.data import IterableDataset

from .gp.nodes.napari_image_source import NapariImageSource
from .meta_data import NapariDatasetMetaData


class NapariDataset(IterableDataset):  # type: ignore
    def __init__(
        self,
        layer: Image,
        axis_names: List[str],
        crop_size: int,
        control_point_spacing: int,
        control_point_jitter: float,
    ):
        """A dataset that serves random samples from a zarr container.

        Args:

            layer:

                The napari layer to use.
                The data should have shape `(s, c, [t,] [z,] y, x)`, where
                `s` = # of samples, `c` = # of channels, `t` = # of frames, and
                `z`/`y`/`x` are spatial extents. The dataset should have an
                `"axis_names"` attribute that contains the names of the used
                axes, e.g., `["s", "c", "y", "x"]` for a 2D dataset.

            axis_names:

                The names of the axes in the napari layer.

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
        self.axis_names = axis_names
        self.control_point_spacing = control_point_spacing
        self.control_point_jitter = control_point_jitter
        self.__read_meta_data()
        self.crop_size = (crop_size,) * self.num_spatial_dims
        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")
        # treat all dimensions as spatial, with a voxel size = 1
        voxel_size = gp.Coordinate((1,) * self.num_dims)
        offset = gp.Coordinate((0,) * self.num_dims)
        shape = gp.Coordinate(self.layer.data.shape)
        raw_spec = gp.ArraySpec(
            roi=gp.Roi(offset, voxel_size * shape),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=voxel_size,
        )
        if self.num_samples == 0:
            self.pipeline = (
                NapariImageSource(
                    self.layer, self.raw, raw_spec, self.spatial_dims
                )
                + gp.Unsqueeze([self.raw], 0)
                + gp.RandomLocation()
            )
        else:
            self.pipeline = (
                NapariImageSource(
                    self.layer, self.raw, raw_spec, self.spatial_dims
                )
                + gp.RandomLocation()
            )

    def __yield_sample(self):
        """An infinite generator of crops."""

        with gp.build(self.pipeline):
            while True:
                # request one sample, all channels, plus crop dimensions
                request = gp.BatchRequest()
                if self.num_channels == 0 and self.num_samples == 0:
                    request[self.raw] = gp.ArraySpec(
                        roi=gp.Roi(
                            (0,) * (self.num_dims),
                            self.crop_size,
                        )
                    )
                elif self.num_channels == 0 and self.num_samples != 0:
                    request[self.raw] = gp.ArraySpec(
                        roi=gp.Roi(
                            (0,) * (self.num_dims), (1, *self.crop_size)
                        )
                    )
                elif self.num_channels != 0 and self.num_samples == 0:
                    request[self.raw] = gp.ArraySpec(
                        roi=gp.Roi(
                            (0,) * (self.num_dims),
                            (self.num_channels, *self.crop_size),
                        )
                    )
                elif self.num_channels != 0 and self.num_samples != 0:
                    request[self.raw] = gp.ArraySpec(
                        roi=gp.Roi(
                            (0,) * (self.num_dims),
                            (1, self.num_channels, *self.crop_size),
                        )
                    )
                sample = self.pipeline.request_batch(request)
                yield sample[self.raw].data[0]

    def __read_meta_data(self):
        meta_data = NapariDatasetMetaData(
            self.layer.data.shape, self.axis_names
        )

        self.num_dims = meta_data.num_dims
        self.num_spatial_dims = meta_data.num_spatial_dims
        self.num_channels = meta_data.num_channels
        self.num_samples = meta_data.num_samples
        self.sample_dim = meta_data.sample_dim
        self.channel_dim = meta_data.channel_dim
        self.time_dim = meta_data.time_dim
        self.spatial_dims = meta_data.spatial_dims

    def get_num_channels(self):
        return 1 if self.num_channels == 0 else self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims
