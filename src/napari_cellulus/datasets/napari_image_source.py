import gunpowder as gp
import numpy as np
from csbdeep.utils import normalize
from gunpowder.array_spec import ArraySpec
from napari.layers import Image


class NapariImageSource(gp.BatchProvider):
    """
    A gunpowder node to pull data from a napari Image
    Args:
        image (Image):
            The napari image layer to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(
        self, image: Image, key: gp.ArrayKey, spec: ArraySpec, spatial_dims
    ):
        self.array_spec = spec

        self.image = gp.Array(
            data=normalize(
                image.data.astype(np.float32), 1, 99.8, axis=spatial_dims
            ),
            spec=spec,
        )
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.image.crop(request[self.key].roi)
        return output
