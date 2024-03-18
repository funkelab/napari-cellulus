from typing import List

import gunpowder as gp
import numpy as np
from napari.layers import Image
from torch.utils.data import IterableDataset

from .meta_data import NapariDatasetMetaData
from .napari_image_source import NapariImageSource


class NapariDataset(IterableDataset):
    def __init__(
        self,
        layer: Image,
        axis_names: List[str],
        crop_size: int,
        density: float,
        kappa: float,
        normalization_factor: float,
    ):
        self.layer = layer
        self.axis_names = axis_names
        self.__read_meta_data()
        self.crop_size = (crop_size,) * self.num_spatial_dims
        self.normalization_factor = normalization_factor
        self.density = density
        self.kappa = kappa
        self.output_shape = tuple(int(_ - 16) for _ in self.crop_size)
        self.normalization_factor = normalization_factor
        self.unbiased_shape = tuple(
            int(_ - (2 * self.kappa)) for _ in self.output_shape
        )
        self.__setup_pipeline()

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
        self.spatial_array = meta_data.spatial_array

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
                    image=self.layer,
                    key=self.raw,
                    spec=raw_spec,
                    spatial_dims=self.spatial_dims,
                )
                + gp.Unsqueeze([self.raw], 0)
                + gp.RandomLocation()
            )
        else:
            self.pipeline = (
                NapariImageSource(
                    image=self.layer,
                    key=self.raw,
                    spec=raw_spec,
                    spatial_dims=self.spatial_dims,
                )
                + gp.RandomLocation()
            )

    def __iter__(self):
        return iter(self.__yield_sample())

    def __yield_sample(self):
        with gp.build(self.pipeline):
            while True:
                array_is_zero = True
                while array_is_zero:
                    # request one sample, all channels, plus crop dimensions
                    request = gp.BatchRequest()
                    if self.num_channels == 0 and self.num_samples == 0:
                        request[self.raw] = gp.ArraySpec(
                            roi=gp.Roi(
                                (0,) * self.num_dims,
                                self.crop_size,
                            )
                        )
                    elif self.num_channels == 0 and self.num_samples != 0:
                        request[self.raw] = gp.ArraySpec(
                            roi=gp.Roi(
                                (0,) * self.num_dims, (1, *self.crop_size)
                            )
                        )
                    elif self.num_channels != 0 and self.num_samples == 0:
                        request[self.raw] = gp.ArraySpec(
                            roi=gp.Roi(
                                (0,) * self.num_dims,
                                (self.num_channels, *self.crop_size),
                            )
                        )
                    elif self.num_channels != 0 and self.num_samples != 0:
                        request[self.raw] = gp.ArraySpec(
                            roi=gp.Roi(
                                (0,) * self.num_dims,
                                (1, self.num_channels, *self.crop_size),
                            )
                        )
                    sample = self.pipeline.request_batch(request)
                    sample_data = sample[self.raw].data[0]
                    # if missing a channel, this is added by the Model class

                    if np.max(sample_data) <= 0.0:
                        pass
                    else:
                        array_is_zero = False
                        (
                            anchor_samples,
                            reference_samples,
                        ) = self.sample_coordinates()
                yield sample_data, anchor_samples, reference_samples

    def get_num_samples(self):
        return self.num_samples

    def get_num_channels(self):
        return self.num_channels

    def get_num_spatial_dims(self):
        return self.num_spatial_dims

    def get_num_dims(self):
        return self.num_dims

    def get_spatial_dims(self):
        return self.spatial_dims

    def get_spatial_array(self):
        return self.spatial_array

    def sample_offsets_within_radius(self, radius, number_offsets):
        if self.num_spatial_dims == 2:
            offsets_x = np.random.randint(
                -radius, radius + 1, size=2 * number_offsets
            )
            offsets_y = np.random.randint(
                -radius, radius + 1, size=2 * number_offsets
            )
            offsets_coordinates = np.stack((offsets_x, offsets_y), axis=1)
        elif self.num_spatial_dims == 3:
            offsets_x = np.random.randint(
                -radius, radius + 1, size=3 * number_offsets
            )
            offsets_y = np.random.randint(
                -radius, radius + 1, size=3 * number_offsets
            )
            offsets_z = np.random.randint(
                -radius, radius + 1, size=3 * number_offsets
            )
            offsets_coordinates = np.stack(
                (offsets_x, offsets_y, offsets_z), axis=1
            )

        in_circle = (offsets_coordinates**2).sum(axis=1) < radius**2
        offsets_coordinates = offsets_coordinates[in_circle]
        not_zero = np.absolute(offsets_coordinates).sum(axis=1) > 0
        offsets_coordinates = offsets_coordinates[not_zero]

        if len(offsets_coordinates) < number_offsets:
            return self.sample_offsets_within_radius(radius, number_offsets)

        return offsets_coordinates[:number_offsets]

    def sample_coordinates(self):
        num_anchors = self.get_num_anchors()
        num_references = self.get_num_references()

        if self.num_spatial_dims == 2:
            anchor_coordinates_x = np.random.randint(
                self.kappa,
                self.output_shape[0] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_y = np.random.randint(
                self.kappa,
                self.output_shape[1] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates = np.stack(
                (anchor_coordinates_x, anchor_coordinates_y), axis=1
            )
        elif self.num_spatial_dims == 3:
            anchor_coordinates_x = np.random.randint(
                self.kappa,
                self.output_shape[0] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_y = np.random.randint(
                self.kappa,
                self.output_shape[1] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates_z = np.random.randint(
                self.kappa,
                self.output_shape[2] - self.kappa + 1,
                size=num_anchors,
            )
            anchor_coordinates = np.stack(
                (
                    anchor_coordinates_x,
                    anchor_coordinates_y,
                    anchor_coordinates_z,
                ),
                axis=1,
            )
        anchor_samples = np.repeat(anchor_coordinates, num_references, axis=0)
        offset_in_pos_radius = self.sample_offsets_within_radius(
            self.kappa, len(anchor_samples)
        )
        reference_samples = anchor_samples + offset_in_pos_radius

        return anchor_samples, reference_samples

    def get_num_anchors(self):
        return int(
            self.density * self.unbiased_shape[0] * self.unbiased_shape[1]
        )

    def get_num_references(self):
        return int(self.density * self.kappa**2 * np.pi)
