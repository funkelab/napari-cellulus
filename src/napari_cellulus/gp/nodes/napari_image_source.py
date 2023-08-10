from typing import Optional

import gunpowder as gp
from gunpowder.array_spec import ArraySpec
from gunpowder.profiling import Timing
from napari.layers import Image


class NapariImageSource(gp.BatchProvider):
    """
    A gunpowder interface to a napari Image
    Args:
        image (Image):
            The napari Image to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(
        self, image: Image, key: gp.ArrayKey, spec: Optional[ArraySpec] = None
    ):
        if spec is None:
            self.array_spec = self._read_metadata(image)
        else:
            self.array_spec = spec
        self.image = gp.Array(
            self._remove_leading_dims(image.data), self.array_spec
        )
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        output[self.key] = self.image.crop(request[self.key].roi)

        timing_provide.stop()

        output.profiling_stats.add(timing_provide)

        return output

    def _remove_leading_dims(self, data):
        while data.shape[0] == 1:
            data = data[0]
        return data

    def _read_metadata(self, image):
        # offset assumed to be in world coordinates
        # TODO: read from metadata
        data_shape = image.data.shape
        # strip leading singleton dimensions (2D data is often given a leading singleton 3rd dimension)
        while data_shape[0] == 1:
            data_shape = data_shape[1:]
        axes = image.metadata.get("axes")
        if axes is not None:
            ndims = len(axes)
            assert ndims <= len(
                data_shape
            ), f"{axes} incompatible with shape: {data_shape}"
        else:
            ndims = len(data_shape)

        offset = gp.Coordinate(image.metadata.get("offset", (0,) * ndims))
        voxel_size = gp.Coordinate(
            image.metadata.get("resolution", (1,) * ndims)
        )
        shape = gp.Coordinate(image.data.shape[-offset.dims :])

        return gp.ArraySpec(
            roi=gp.Roi(offset, voxel_size * shape),
            dtype=image.dtype,
            interpolatable=True,
            voxel_size=voxel_size,
        )
